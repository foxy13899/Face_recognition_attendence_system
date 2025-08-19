
# Face Recognition Attendance System (BY FOXY!!!)
# abe ye download kar diyo be: opencv-python face_recognition numpy pandas (python 3.11 works great)

import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

# === Step 1: Load Known Faces ===
known_faces = []
known_names = []

dataset_path = "dataset"  # Folder with subfolders per person

for name in os.listdir(dataset_path):
    print(name)
    person_path = os.path.join(dataset_path, name)
    if not os.path.isdir(person_path):
        continue
    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            known_faces.append(encoding[0])
            known_names.append(name)

print(f"Loaded encodings for {len(known_faces)} faces")

# === Step 2: Attendance CSV ===
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# === Step 3: Start Camera ===
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names_in_frame = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[best_match_index]:
            name = known_names[best_match_index]

            if name not in names_in_frame:
                names_in_frame.append(name)

                # === Step 4: Mark Attendance ===
                df = pd.read_csv(attendance_file)
                now = datetime.now()
                date = now.strftime("%d-%m-%Y")
                time = now.strftime("%H:%M:%S")

                # Prevent duplicate entries for same day
                if not ((df["Name"] == name) & (df["Date"] == date)).any():
                    new_row = {"Name": name, "Date": date, "Time": time}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    df.to_csv(attendance_file, index=False)
                    print(f"Attendance marked for {name} at {time}")

        # Draw box around face
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

