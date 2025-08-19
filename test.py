# Face Recognition Attendance System (BY FOXY!!!)
# abe ye download kar diyo be: opencv-python face_recognition numpy pandas (python 3.11 works great)

import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# =============================
# Config
# =============================
DATASET_DIR = 'dataset'
CAMERA_INDEX = 0
FRAME_RESIZE_SCALE = 0.25   # 0.25 = 1/4 size for speed
TOLERANCE = 0.55            # lower = stricter match (0.4-0.6 typical)
SIDEBAR_WIDTH = 250
FONT = cv2.FONT_HERSHEY_SIMPLEX

# =============================
# Helpers
# =============================

def load_known_faces(dataset_dir: str):
    """Load images from dataset_dir/<person>/*.jpg and compute encodings."""
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"[WARN] Dataset directory '{dataset_dir}' did not exist. Created it.")
        return [], []

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    encodings, names = [], []

    people = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
    print(f"[INFO] Found {len(people)} person folder(s): {people}")

    for person in people:
        person_dir = os.path.join(dataset_dir, person)
        files = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in image_exts]
        if not files:
            print(f"[WARN] No image files in {person_dir}.")
            continue
        for f in files:
            path = os.path.join(person_dir, f)
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not read image: {path}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_encodings(rgb)
            if faces:
                encodings.append(faces[0])
                names.append(person)
    return encodings, names

def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    filename = f'attendance_{date_string}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

    if not ((df['Name'] == name) & (df['Date'] == date_string)).any():
        df.loc[len(df)] = [name, date_string, time_string]
        df.to_csv(filename, index=False)

# =============================
# Main
# =============================

def main():
    known_encodings, known_names = load_known_faces(DATASET_DIR)
    print(f"[INFO] Encoded {len(known_encodings)} faces.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    present_students = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for enc, loc in zip(encodings, locations):
            distances = face_recognition.face_distance(known_encodings, enc)
            if len(distances) == 0:
                continue
            best_idx = np.argmin(distances)
            if distances[best_idx] < TOLERANCE:
                name = known_names[best_idx].upper()
            else:
                name = "UNKNOWN"

            # Scale back up face location
            y1, x2, y2, x1 = [v * int(1/FRAME_RESIZE_SCALE) for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), FONT, 1, (255, 255, 255), 2)

            if name != "UNKNOWN" and name not in present_students:
                present_students.add(name)
                mark_attendance(name)

        # Sidebar overlay
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w - SIDEBAR_WIDTH, 0), (w - 50 , h - 100), (255, 255, 255), 1)
        cv2.putText(frame, "Present:", (w - SIDEBAR_WIDTH + 10, 30), FONT, 0.7, (0, 0, 0), 2)

        y_offset = 60
        for n in sorted(present_students):
            cv2.putText(frame, f"- {n}", (w - SIDEBAR_WIDTH + 10, y_offset), FONT, 0.6, (0, 128, 0), 2)
            y_offset += 30

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
