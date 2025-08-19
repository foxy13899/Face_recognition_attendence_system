# Face_recognition_attendence, BY FOXY!!!
👇

🎓 Face Recognition Attendance System

Uses Python, OpenCV, and face_recognition to mark student attendance in real-time using a webcam or Raspberry Pi camera.

✨ Features

Detects multiple faces in a video feed simultaneously

Matches faces with a dataset of known students

Marks attendance automatically in a CSV file (attendance_YYYY-MM-DD.csv)

Displays a live sidebar showing who is present

Runs on laptop webcams or Raspberry Pi + Pi Camera

🛠️ Tech Stack

Python 3.x

OpenCV
 – image & video processing

face_recognition
 – deep learning based face encodings

NumPy & Pandas – data handling and CSV storage

📂 Project Structure
├── dataset/             # Training images (one folder per person)
│   ├── Alice/           # Example student folder
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Bob/
│       └── img1.jpg
├── attendance_YYYY-MM-DD.csv   # Auto-generated daily attendance file
├── main.py              # Main face recognition + attendance code
└── README.md            # Project documentation

📸 How It Works

Place student images inside dataset/, one folder per student (e.g., dataset/Alice/)

Run the program:

python main.py


The camera feed opens. Faces are detected and recognised.

Attendance is logged into a CSV file for the current date.

Press Q to quit.

⚡ Setup & Installation

Clone the repository:

git clone https://github.com/foxy13899/face-attendance-system.git
cd face-attendance-system


Install dependencies:

pip install opencv-python face_recognition numpy pandas


⚠️ On Raspberry Pi, installing dlib (required by face_recognition) may take time to compile.

Run the script:

python main.py

🖥️ Running on Raspberry Pi

Attach the Pi Camera Module or USB webcam

Enable the camera in Raspberry Pi config

Install dependencies:

sudo apt-get update
sudo apt-get install python3-opencv
pip3 install face_recognition numpy pandas


Run:

python3 main.py

📊 Example Attendance Output
Name, Date, Time
ALICE,2025-08-19,09:15:32
BOB,2025-08-19,09:17:08

🎯 Project Goals

Show how AI + hardware can solve real classroom problems
Create a base for future IoT integration (automatic door unlock, cloud sync, etc.)

👩‍💻 Contributors
Shridhar Arora
