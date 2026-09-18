# Emotion Attendance System

A Python-based machine learning project that combines:

- Face detection using OpenCV
- Student face recognition using LBPH
- Facial expression classification using a CNN
- Student registration and automatic face-image collection
- Attendance recording in a CSV file
- Webcam-based live demonstration

> **Important:** This project is intended as an educational/demo system. Facial-expression recognition is imperfect and should not be used as the sole basis for real attendance, access, disciplinary, or other consequential decisions.

---

## 1. Project Structure

After copying the project to another laptop, the folder should look approximately like this:

```text
emotion_attendance/
│
├── dataset/
│   ├── happy/
│   ├── neutral/
│   └── sad/
│
├── students/
│   ├── 101/
│   └── ...
│
├── model/
│   ├── emotion_model.keras
│   ├── face_model.yml
│   └── students.json
│
├── collect_data.py
├── train_emotion.py
├── register_student.py
├── train_model.py
├── emotion_attendance.py
├── attendance.csv
└── requirements.txt
```

If you are giving this project to another person, include the trained `model/` folder and the `students/` folder if you want the same trained students to be recognized.

---

## 2. Requirements

### Hardware

- Windows laptop/PC
- Working webcam
- At least 4 GB RAM recommended
- Internet connection for the initial Python/package installation

### Software

- Windows 10 or Windows 11
- Python 3.11 (64-bit) recommended
- Webcam drivers installed

Python can be downloaded from the official Python website:

https://www.python.org/downloads/windows/

---

## 3. Install Python

Install Python 3.11 64-bit.

During installation, make sure to select:

```text
Add python.exe to PATH
```

After installation, open a new Command Prompt and check:

```bat
python --version
```

It should show something similar to:

```text
Python 3.11.x
```

If Windows uses the Python launcher, you can also check:

```bat
py -3.11 --version
```

---

## 4. Copy the Project

Copy the complete project folder to the new laptop.

For example:

```text
D:\emotion_attendance
```

or:

```text
C:\emotion_attendance
```

Open Command Prompt in that folder.

Example:

```bat
cd /d D:\emotion_attendance
```

---

## 5. Create a Virtual Environment

Create a new virtual environment:

```bat
python -m venv venv
```

Activate it:

```bat
venv\Scripts\activate
```

You should now see:

```text
(venv)
```

at the beginning of the command prompt.

Example:

```text
(venv) D:\emotion_attendance>
```

---

## 6. Install Python Packages

Upgrade pip:

```bat
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bat
python -m pip install -r requirements.txt
```

### If `requirements.txt` gives a TensorFlow version error

Do not randomly change TensorFlow versions if the project already has a working environment.

First check:

```bat
python --version
python -m pip show tensorflow
python -m pip show opencv-contrib-python
python -m pip show numpy
python -m pip show Pillow
```

The safest way to create a requirements file from a working copy is:

```bat
python -m pip freeze > requirements.txt
```

Then copy that `requirements.txt` with the project.

---

## 7. Verify OpenCV

Run:

```bat
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier')); print(hasattr(cv2, 'face')); print(hasattr(cv2.face, 'LBPHFaceRecognizer_create'))"
```

The final three values should be:

```text
True
True
True
```

The important part is that `cv2.face.LBPHFaceRecognizer_create` exists.

If it does not, install the contrib package:

```bat
python -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
python -m pip install opencv-contrib-python==4.10.0.84
```

Then run the verification command again.

---

## 8. Verify TensorFlow

Run:

```bat
python -c "import tensorflow as tf; print(tf.__version__)"
```

A TensorFlow version number should be displayed.

Also test Pillow:

```bat
python -c "from PIL import Image; print('Pillow OK')"
```

---

# 9. Running the Project

There are two different situations.

## A. Run the project with the existing trained models

If you copied the trained:

```text
model/
```

folder and student data, you do NOT need to train again.

Simply activate the virtual environment:

```bat
venv\Scripts\activate
```

Then run:

```bat
python emotion_attendance.py
```

The program should load:

```text
model/emotion_model.keras
model/face_model.yml
model/students.json
```

The webcam window should open.

Press:

```text
Q
```

to close the application.

---

# 10. Register a New Student

To add a new student:

```bat
python register_student.py
```

Enter:

```text
Student ID: 102
Student Name: Rahul
```

The program automatically creates:

```text
students\102
```

and captures face images.

You do not need to manually create the student folder.

---

# 11. Train the Face Recognition Model

After adding a new student, retrain the face recognition model:

```bat
python train_model.py
```

The program reads the images from:

```text
students/
```

and creates/updates:

```text
model\face_model.yml
```

Example:

```text
Loading student: 101
Images: 104

Loading student: 102
Images: 100

Training face recognition model...

TRAINING COMPLETED
Model saved: model\face_model.yml
```

### Important

You only need to retrain the **face recognition model** when adding or changing students.

You do NOT need to retrain the emotion model just because you added a new student.

---

# 12. Emotion Model Training

The emotion model is trained separately using the emotion dataset:

```text
dataset/
├── happy/
├── neutral/
└── sad/
```

If you need to retrain the emotion model, run the emotion-training script used by your project, for example:

```bat
python train_emotion.py
```

It should create:

```text
model\emotion_model.keras
```

If your project uses a different filename for the emotion-training script, use that script instead.

---

# 13. Live Attendance Demo

Start the complete system:

```bat
python emotion_attendance.py
```

The system performs:

```text
Webcam
   ↓
Face Detection
   ↓
Student Face Recognition
   ↓
Emotion Recognition
   ↓
Status
   ↓
Attendance CSV
```

Example display:

```text
Student: Hitesh
ID: 101
Emotion: happy
Confidence: 85.4%
ATTENDANCE DEMO
```

The system records a successful demo attendance entry in:

```text
attendance.csv
```

---

# 14. Attendance CSV

The attendance file contains columns similar to:

```text
Date
Time
Student ID
Student Name
Emotion
Confidence
Status
```

Example:

```csv
Date,Time,Student ID,Student Name,Emotion,Confidence,Status
2026-09-18,10:30:15,101,Hitesh,happy,85.42,ATTENDANCE DEMO
```

You can open `attendance.csv` using Microsoft Excel.

---

# 15. Recommended Run Order

### First-time setup

```bat
cd /d D:\emotion_attendance
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then verify OpenCV and TensorFlow.

### If trained models are already included

```bat
python emotion_attendance.py
```

### If adding a new student

```bat
python register_student.py
python train_model.py
python emotion_attendance.py
```

### If retraining the emotion model

Use the project's emotion-training script:

```bat
python train_emotion.py
```

Then run:

```bat
python emotion_attendance.py
```

---

# 16. Common Problems

## Problem: `python is not recognized`

Install Python 3.11 and enable:

```text
Add Python to PATH
```

Then reopen Command Prompt.

---

## Problem: Webcam does not open

Make sure:

- The webcam is connected.
- Windows Camera permissions allow desktop apps to access the camera.
- No other application is currently using the webcam.

You can also try changing:

```python
cv2.VideoCapture(0)
```

to:

```python
cv2.VideoCapture(1)
```

if the laptop has multiple cameras.

---

## Problem: `cv2.face` does not exist

Install:

```bat
python -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
python -m pip install opencv-contrib-python==4.10.0.84
```

Then test:

```bat
python -c "import cv2; print(hasattr(cv2.face, 'LBPHFaceRecognizer_create'))"
```

It should print:

```text
True
```

---

## Problem: `No module named tensorflow`

Activate the virtual environment:

```bat
venv\Scripts\activate
```

Then:

```bat
python -m pip install tensorflow
```

Or install using:

```bat
python -m pip install -r requirements.txt
```

---

## Problem: `No module named PIL`

Install Pillow:

```bat
python -m pip install Pillow
```

---

## Problem: Student is shown as `Unknown`

Check that:

```text
model\face_model.yml
```

exists.

If a new student was registered, retrain:

```bat
python train_model.py
```

Also make sure the student's folder exists:

```text
students\101
students\102
...
```

---

## Problem: Model file not found

Check that these files exist:

```text
model\emotion_model.keras
model\face_model.yml
model\students.json
```

If they are missing, copy them from the original working project or train the corresponding model.

---

# 17. Important Notes for Demonstration

For a college/project demonstration:

1. Use good lighting.
2. Keep the face reasonably centered in the webcam.
3. Register enough face images for each student.
4. Use different head positions while registering.
5. Test each registered student separately.
6. Keep the trained model files with the project.
7. Keep a backup of `attendance.csv`.

The emotion classifier is a machine-learning demonstration and can make incorrect predictions. Do not treat the predicted emotion as a definitive measurement of a person's actual emotional state.

---

# 18. Quick Commands

### Activate environment

```bat
venv\Scripts\activate
```

### Start application

```bat
python emotion_attendance.py
```

### Register student

```bat
python register_student.py
```

### Train face model

```bat
python train_model.py
```

### Train emotion model

```bat
python train_emotion.py
```

### Check Python

```bat
python --version
```

### Check installed packages

```bat
python -m pip list
```

### Generate requirements from a working environment

```bat
python -m pip freeze > requirements.txt
```

---

## 19. Project Workflow

```text
              ┌──────────────┐
              │    Webcam    │
              └──────┬───────┘
                     ↓
             ┌───────────────┐
             │ Face Detection│
             └───────┬───────┘
                     ↓
          ┌─────────────────────┐
          │ Student Recognition │
          │       LBPH          │
          └──────────┬──────────┘
                     ↓
          ┌─────────────────────┐
          │ Emotion Recognition │
          │       CNN           │
          └──────────┬──────────┘
                     ↓
       ┌────────────────────────────┐
       │ Happy / Neutral / Sad      │
       └──────────────┬─────────────┘
                      ↓
              ┌──────────────┐
              │ Demo Status  │
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │ attendance   │
              │    .csv      │
              └──────────────┘
```

---

## 20. Final Command

After setup, the normal command to start the project is:

```bat
python emotion_attendance.py
```

Press `Q` to exit the webcam window.
