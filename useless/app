import os
import cv2
import sqlite3
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from scipy.spatial.distance import cosine

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------- Paths -----------------
DB_PATH = "test.db"
IMAGES_DIR = "test_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# ----------------- Database -----------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

# Create users table
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)
""")
conn.commit()
conn.close()

# ----------------- Haar Cascade -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------- ORB Recognition -----------------
# def recognize_face(frame, threshold=0.5):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
#     results = []

#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]

#         orb = cv2.ORB_create()
#         kp1, des1 = orb.detectAndCompute(face, None)
#         if des1 is None or len(kp1) == 0:
#             continue

#         recognized_name = "Unknown"
#         min_distance = float('inf')

#         # Loop through each registered user
#         for user_name in os.listdir(IMAGES_DIR):
#             user_dir = os.path.join(IMAGES_DIR, user_name)
#             for img_file in os.listdir(user_dir):
#                 stored_face = cv2.imread(os.path.join(user_dir, img_file), cv2.IMREAD_GRAYSCALE)
#                 kp2, des2 = orb.detectAndCompute(stored_face, None)
#                 if des2 is None or len(kp2) == 0:
#                     continue

#                 # Compute descriptors distance using cosine
#                 emb1 = np.mean(des1, axis=0)
#                 emb1 = emb1 / np.linalg.norm(emb1)
#                 emb2 = np.mean(des2, axis=0)
#                 emb2 = emb2 / np.linalg.norm(emb2)
#                 dist = cosine(emb1, emb2)

#                 if dist < threshold and dist < min_distance:
#                     min_distance = dist
#                     recognized_name = user_name

#         results.append({
#             "x": int(x),
#             "y": int(y),
#             "w": int(w),
#             "h": int(h),
#             "name": recognized_name
#         })

#     return results
def recognize_face(frame, threshold=0.5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(face, None)
        if des1 is None or len(kp1) == 0:
            results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "name": "Unknown"})
            continue

        emb1 = np.mean(des1, axis=0)
        emb1 = emb1 / np.linalg.norm(emb1)

        recognized_name = "Unknown"
        min_distance = threshold  # Only assign name if distance < threshold

        # Loop through registered users
        for user_name in os.listdir(IMAGES_DIR):
            user_dir = os.path.join(IMAGES_DIR, user_name)
            for img_file in os.listdir(user_dir):
                stored_face = cv2.imread(os.path.join(user_dir, img_file), cv2.IMREAD_GRAYSCALE)
                kp2, des2 = orb.detectAndCompute(stored_face, None)
                if des2 is None or len(kp2) == 0:
                    continue

                emb2 = np.mean(des2, axis=0)
                emb2 = emb2 / np.linalg.norm(emb2)
                dist = cosine(emb1, emb2)

                if dist < min_distance:
                    min_distance = dist
                    recognized_name = user_name

        results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "name": recognized_name})

    return results



# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/register_face', methods=['POST'])
def register_face():
    data = request.get_json()
    name = data.get('name')
    images = data.get('images')

    if not name or not images or len(images) == 0:
        return jsonify({"message": "Name and images are required"}), 400

    try:
        person_dir = os.path.join(IMAGES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        saved_count = 0
        idx = 0
        while saved_count < 10 and idx < len(images):
            img_data = images[idx]
            img_str = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_bytes))
            frame = np.array(img.convert('RGB'))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
            if len(faces) == 0:
                idx += 1
                continue

            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            face_img = Image.fromarray(face)
            face_img.save(os.path.join(person_dir, f"{saved_count+1}.jpg"))

            saved_count += 1
            idx += 1

        if saved_count < 10:
            return jsonify({"message": f"Only {saved_count} faces saved. Need 10 valid snapshots."}), 400

        # Add name to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Successfully saved 10 face images for {name}!"})

    except Exception as e:
        return jsonify({"message": str(e)}), 500

# ----------------- Live Recognition -----------------
# previous_name = None
previous_names = {}  # key: face index, value: {"name": name, "count": count}
STABLE_FRAMES = 3   # number of frames name must persist

@socketio.on('frame')
def handle_frame(data):
    global previous_name
    img_data = data.get('image')
    if not img_data:
        emit('faces', {"faces":[]})
        return

    try:
        img_str = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))
        frame = np.array(img.convert('RGB'))

        results = recognize_face(frame, threshold=0.5)
        # emit('faces', {"faces": results})

        # Smooth names to reduce flicker
        smoothed_results = []
        for idx, f in enumerate(results):
            prev = previous_names.get(idx, {"name": f["name"], "count": 0})
            if f["name"] == prev["name"]:
                prev["count"] += 1
            else:
                prev["count"] = 0
                prev["name"] = f["name"]

            # Only update name if stable for STABLE_FRAMES
            if prev["count"] >= STABLE_FRAMES:
                smoothed_results.append(f)
            else:
                smoothed_results.append({**f, "name": prev["name"]})

            previous_names[idx] = prev

        emit('faces', {"faces": smoothed_results})

    except Exception as e:
        print(f"⚠️ Live recognition error: {e}")
        emit('faces', {"faces":[]})

# ----------------- Run App -----------------
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)

























@app.route('/register_face', methods=['POST'])
def register_face():
    data = request.get_json()
    name = data.get('name')
    images = data.get('images')  # Expect a list of 10 base64 images

    if not name or not images or len(images) < 10:
        return jsonify({"message": "Name or at least 10 images required"}), 400

    try:
        embeddings = []
        for idx, img_data in enumerate(images[:10]):
            img_str = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_bytes))
            frame = np.array(img.convert('RGB'))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
            if len(faces) == 0:
                # Fallback: use center crop
                h, w = gray.shape
                cx, cy = w//2, h//2
                size = min(w, h)//2
                face = gray[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
            else:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]


            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(face, None)
            # if des is None or len(kp)==0:
            #     continue
            if des is None or len(kp)==0:
                # Fallback: create a random embedding (or repeat previous)
                embedding = np.random.rand(32)
            else:
                embedding = np.mean(des, axis=0)

            embedding = np.mean(des, axis=0)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        if len(embeddings) == 0:
            return jsonify({"message": "No valid faces detected"}), 400

        # Store embeddings in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        for emb in embeddings:
            cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, pickle.dumps(emb)))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Face for {name} registered successfully with {len(embeddings)} snapshots!"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500