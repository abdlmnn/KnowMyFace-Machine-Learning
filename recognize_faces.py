import cv2
import sqlite3
import numpy as np
import pickle
from scipy.spatial.distance import cosine

# Connect to SQLite
conn = sqlite3.connect("dataset.db")
cursor = conn.cursor()

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to get the closest match from the database
def recognize_face(embedding):
    cursor.execute("SELECT id, name, embedding FROM faces")
    rows = cursor.fetchall()

    min_distance = float('inf')
    recognized_name = None

    # Compare the captured embedding with all stored embeddings
    for row in rows:
        stored_embedding = pickle.loads(row[2])  # Deserialize the embedding
        distance = cosine(embedding, stored_embedding)

        # Update the recognized name if a closer match is found
        if distance < min_distance:
            min_distance = distance
            recognized_name = row[1]  # Store the name of the recognized person

    return recognized_name, min_distance

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("📸 Face recognition started... Press 'q' to quit.")

previous_name = None  # Track the previously recognized name

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame. Try again.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Loop through each face detected
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Feature extraction with ORB (same as in capture_faces.py)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray_face, None)

        if descriptors is not None and len(keypoints) > 0:
            embedding = np.mean(descriptors, axis=0)

            # Recognize the face based on the embedding
            recognized_name, distance = recognize_face(embedding)

            if recognized_name is not None and distance < 0.5:  # Threshold for recognition
                # Prevent the same name from being repeatedly recognized in a short time frame
                if recognized_name != previous_name:
                    previous_name = recognized_name
                    cv2.putText(frame, f"{recognized_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{recognized_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("❌ Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
