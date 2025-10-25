import os
import cv2
import sqlite3
import numpy as np
import io
import base64
import pickle
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from scipy.spatial.distance import cosine

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------- Database -----------------
DB_PATH = "test.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

# Create embeddings table
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()
conn.close()

# ----------------- Haar Cascade -----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------- ORB Helper -----------------
orb = cv2.ORB_create()

def compute_embedding(face_img):
    """Compute ORB-based embedding (average descriptor vector)."""
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray_face, None)
    if des is None or len(kp) == 0:
        return None
    emb = np.mean(des, axis=0)
    return emb / np.linalg.norm(emb)

# ----------------- Registration -----------------
@app.route("/register_face", methods=["POST"])
def register_face():
    data = request.get_json()
    name = data.get("name")
    images = data.get("images")

    if not name or not images:
        return jsonify({"message": "Name and images required"}), 400

    embeddings = []
    saved_count = 0
    idx = 0

    while saved_count < 5 and idx < len(images):
        try:
            img_str = images[idx].split(",")[1]
            img_bytes = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_bytes))
            frame = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) == 0:
                idx += 1
                continue

            x, y, w, h = faces[0]
            face_crop = frame[y:y + h, x:x + w]

            embedding = compute_embedding(face_crop)
            if embedding is not None:
                embeddings.append(embedding)
                saved_count += 1

        except Exception as e:
            print(f"Registration error: {e}")
        idx += 1

    if saved_count < 5:
        return jsonify({"message": f"Only {saved_count}/5 faces saved. Try again"}), 400

    # Save multiple embeddings (delete old ones first)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE name=?", (name,))
    for emb in embeddings:
        cursor.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)",
            (name, pickle.dumps(emb)),
        )
    conn.commit()
    conn.close()

    return jsonify({"message": f"Registered {name} with {saved_count} faces"})

# ----------------- Recognition -----------------
def recognize_face(frame, threshold=0.32):
    """Recognize faces in a frame using ORB embeddings and cosine similarity.
       Returns bounding box, name, confidence, and embedding for smoothing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    results = []

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    # Group embeddings by user
    db_faces = {}
    for stored_name, stored_emb in rows:
        emb = pickle.loads(stored_emb)
        db_faces.setdefault(stored_name, []).append(emb)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        emb1 = compute_embedding(face_crop)

        if emb1 is None:
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "name": "Unknown", "confidence": 1.0, "embedding": None
            })
            continue

        recognized_name = "Unknown"
        min_dist = 1.0

        # Compare with all embeddings of each user
        for name, embs in db_faces.items():
            dists = [cosine(emb1, e) for e in embs if e is not None]
            if not dists:
                continue
            avg_dist = np.mean(dists)
            if avg_dist < threshold and avg_dist < min_dist:
                min_dist = avg_dist
                recognized_name = name

        results.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "name": recognized_name,
            "confidence": round(min_dist, 3),
            "embedding": emb1  # pass embedding for smoothing
        })

    return results


# ----------------- Live Recognition -----------------
STABLE_FRAMES = 3
previous_faces = {}  # face_id -> {"name": ..., "count": ..., "embedding": ..., "bbox": ...}
IOU_THRESHOLD = 0.3

def iou(boxA, boxB):
    # Compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / (boxAArea + boxBArea - interArea)

@socketio.on("frame")
def handle_frame(data):
    img_data = data.get("image")
    if not img_data:
        emit("faces", {"faces": []})
        return

    try:
        img_str = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))
        frame = np.array(img.convert("RGB"))

        results = recognize_face(frame)

        smoothed = []
        updated_faces = {}

        # Match current detections with previous_faces using IoU
        for idx, f in enumerate(results):
            nx, ny, nw, nh = f["x"], f["y"], f["w"], f["h"]
            nbox = [nx, ny, nx+nw, ny+nh]

            best_iou = 0
            best_id = None
            for face_id, prev in previous_faces.items():
                pbox = prev["bbox"]
                iou_val = iou(nbox, pbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = face_id

            if best_iou > IOU_THRESHOLD:
                # matched previous face
                prev = previous_faces[best_id]
                # Update embedding moving average
                if "embedding" in f:
                    prev_emb = prev.get("embedding")
                    if prev_emb is not None:
                        f_emb = f["embedding"]
                        prev["embedding"] = 0.7 * prev_emb + 0.3 * f_emb
                # Update name based on confidence
                if f["confidence"] < 0.32 and f["name"] == prev["name"]:
                    prev["count"] += 1
                else:
                    if f["confidence"] < 0.32:
                        prev["name"] = f["name"]
                        prev["count"] = 1
                    else:
                        prev["count"] -= 1
                        if prev["count"] <= 0:
                            prev["name"] = "Unknown"
                            prev["count"] = 0
                stable_name = prev["name"] if prev["count"] >= STABLE_FRAMES else "Unknown"
                prev["bbox"] = nbox
                smoothed.append({
                    "x": nx,
                    "y": ny,
                    "w": nw,
                    "h": nh,
                    "name": stable_name,
                    "confidence": f["confidence"]  # JSON-safe
                })

                updated_faces[best_id] = prev
            else:
                # new face
                updated_faces[idx] = {
                    "name": f["name"] if f["confidence"] < 0.32 else "Unknown",
                    "count": 1 if f["confidence"] < 0.32 else 0,
                    "embedding": f.get("embedding"),
                    "bbox": nbox
                }

                smoothed.append({
                    "x": nx,
                    "y": ny,
                    "w": nw,
                    "h": nh,
                    "name": updated_faces[idx]["name"],
                    "confidence": f["confidence"]
                })

        previous_faces.clear()
        previous_faces.update(updated_faces)
        emit("faces", {"faces": smoothed})

    except Exception as e:
        print(f"Live recognition error: {e}")
        emit("faces", {"faces": []})

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/recognition")
def recognition():
    return render_template("recognition.html")

# ----------------- Run -----------------
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5001))
#     socketio.run(app, host="0.0.0.0", port=port, debug=False)
