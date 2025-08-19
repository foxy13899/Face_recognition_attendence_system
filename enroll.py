# enrollment.py
import cv2
import os

dataset_path = "dataset"

person_name = input("Enter name of person to enroll: ")
save_path = os.path.join(dataset_path, person_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(0)
frame_count = 0

print("Recording... press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Enrollment", frame)

    # Save every 5th frame for variety
    if frame_count % 5 == 0:
        img_path = os.path.join(save_path, f"{person_name}_{frame_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count > 100:
        break  # Stop after ~100 frames (~10 sec)

cap.release()
cv2.destroyAllWindows()
print("Enrollment complete.")
