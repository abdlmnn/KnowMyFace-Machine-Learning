import os
import cv2
import sqlite3
import numpy as np
import pickle
import time

# Connect to SQLite
conn = sqlite3.connect("dataset.db")
cursor = conn.cursor()

# Create a table to store face embeddings if it doesn't exist
cursor.execute(""" 
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Get user name for this set of snaps
    name = input("Enter your name: ").strip()

    # Create a folder for this person to store their face images
    data_folder = "dataset"
    person_folder = os.path.join(data_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    false_positives = 0
    true_negatives = 0
    total_frames = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    snap_count = 0
    embeddings = []

    print("📸 Get ready for 5 face snaps... Press 's' to take a snap. Press 'q' to quit.")

    while snap_count < 5:
        # Start time for processing the frame
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Try again.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Debugging: Print how many faces were detected
        print(f"Faces detected: {len(faces)}")
        # Measure False Positives and True Negatives
        if len(faces) == 0:
            true_negatives += 1  # Correctly identified no faces
        else:
            false_positives += 1  # Incorrectly detected faces

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:  # Allow a slight variation in aspect ratio
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face detected! Press 's'", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Possible eye detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Snap Face - Press 's'", frame)
        key = cv2.waitKey(1)

        if key == ord('s') and len(faces) > 0:
            try:
                # Extract the region of the first face detected
                (x, y, w, h) = faces[0]
                face = frame[y:y + h, x:x + w]

                # Display the face crop for debugging
                cv2.imshow("Captured Face", face)
                cv2.waitKey(1)

                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                orb = cv2.ORB_create()
                keypoints, descriptors = orb.detectAndCompute(gray_face, None)

                if descriptors is not None and len(keypoints) > 0:
                    # Simple averaging of descriptors as embedding
                    embedding = np.mean(descriptors, axis=0)

                    # Save the face image in the corresponding folder for this person
                    image_filename = os.path.join(person_folder, f"snap_{snap_count + 1}.jpg")
                    cv2.imwrite(image_filename, face)  # Save the face as an image file
                    print(f"Face image saved as: {image_filename}")

                    # Add the embedding to the list of embeddings
                    embeddings.append(embedding)
                    snap_count += 1
                    print(f"✅ Snap {snap_count}/5 captured.")
                else:
                    print("⚠️ Not enough keypoints detected for face embedding. Try again.")
            except Exception as e:
                print(f"⚠️ Face representation failed. Try again. Error: {e}")

        elif key == ord('q'):
            print("❌ Exiting.")
            break

        # Calculate Processing Speed
        end_time = time.time()
        processing_speed = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Processing Speed: {processing_speed:.2f} ms per frame")

    cap.release()
    cv2.destroyAllWindows()

    # Store embeddings in the database after 5 snaps
    if snap_count == 5:
        for embedding in embeddings:
            cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, pickle.dumps(embedding)))
        conn.commit()
        print("✅ All 5 face snaps saved and embeddings stored in the database.")
    else:
        print(f"⚠️ Only {snap_count} face(s) saved.")

    # Calculate False Positive Rate (FPR) after all frames
    total_negatives = false_positives + true_negatives
    if total_negatives > 0:
        false_positive_rate = false_positives / total_negatives
        print(f"False Positive Rate: {false_positive_rate:.2f}")
    else:
        print("Not enough data to calculate FPR.")

    # Ask the user if they want to continue or quit
    continue_input = input("Do you want to add another person? (y/n): ").strip().lower()
    if continue_input != 'y':
        break

conn.close()
