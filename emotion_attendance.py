import cv2
import os
import json
import csv
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# SETTINGS
# ==========================================

EMOTION_MODEL = "model/emotion_model.keras"
FACE_MODEL = "model/face_model.yml"
STUDENTS_FILE = "model/students.json"
ATTENDANCE_FILE = "attendance.csv"

EMOTIONS = ["happy", "neutral", "sad"]

EMOTION_THRESHOLD = 60
FACE_THRESHOLD = 80

# ==========================================
# LOAD EMOTION MODEL
# ==========================================

print("Loading emotion model...")

emotion_model = load_model(EMOTION_MODEL)

print("Emotion model loaded.")

# ==========================================
# LOAD FACE RECOGNITION MODEL
# ==========================================

print("Loading face recognition model...")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(FACE_MODEL)

print("Face recognition model loaded.")

# ==========================================
# LOAD STUDENT INFORMATION
# ==========================================

if os.path.exists(STUDENTS_FILE):

    with open(STUDENTS_FILE, "r") as file:
        students = json.load(file)

else:

    print("ERROR: students.json not found.")
    print("Please run register_student.py first.")
    exit()

# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# ATTENDANCE FILE
# ==========================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Student ID",
            "Student Name",
            "Emotion",
            "Confidence",
            "Status"
        ])

# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Cannot open webcam.")
    exit()

print()
print("==============================")
print("EMOTION ATTENDANCE SYSTEM")
print("==============================")
print("Press Q to quit.")
print()

# Prevent duplicate attendance
recorded_students = set()

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read webcam.")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in detected_faces:

        # ==================================
        # FACE RECOGNITION
        # ==================================

        face_gray = gray[y:y+h, x:x+w]

        face_gray = cv2.resize(
            face_gray,
            (200, 200)
        )

        student_id, face_distance = face_recognizer.predict(
            face_gray
        )

        # Default student information
        student_name = "Unknown"
        recognized = False

        # Lower LBPH distance = better match
        if (
            str(student_id) in students
            and face_distance < FACE_THRESHOLD
        ):

            student_id_str = str(student_id)
            student_name = students[student_id_str]
            recognized = True

        else:

            student_id_str = "Unknown"

        # ==================================
        # EMOTION RECOGNITION
        # ==================================

        emotion_face = gray[y:y+h, x:x+w]

        emotion_face = cv2.resize(
            emotion_face,
            (48, 48)
        )

        emotion_face = emotion_face.astype(
            "float32"
        ) / 255.0

        emotion_face = np.expand_dims(
            emotion_face,
            axis=0
        )

        emotion_face = np.expand_dims(
            emotion_face,
            axis=-1
        )

        predictions = emotion_model.predict(
            emotion_face,
            verbose=0
        )[0]

        emotion_index = np.argmax(
            predictions
        )

        emotion = EMOTIONS[emotion_index]

        emotion_confidence = (
            predictions[emotion_index] * 100
        )

        # ==================================
        # STATUS
        # ==================================

        if not recognized:

            status = "UNKNOWN STUDENT"

        elif emotion_confidence < EMOTION_THRESHOLD:

            status = "CHECK CONDITION"

        elif emotion == "happy":

            status = "ATTENDANCE DEMO"

        elif emotion == "neutral":

            status = "YOU MAY SIT"

        elif emotion == "sad":

            status = "DEMO: NOT MARKED"

        else:

            status = "CHECK CONDITION"

        # ==================================
        # ATTENDANCE RECORD
        # ==================================

        if (
            recognized
            and emotion == "happy"
            and emotion_confidence >= EMOTION_THRESHOLD
            and student_id_str not in recorded_students
        ):

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            with open(
                ATTENDANCE_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    date,
                    time,
                    student_id_str,
                    student_name,
                    emotion,
                    f"{emotion_confidence:.2f}",
                    status
                ])

            recorded_students.add(
                student_id_str
            )

            print()
            print("==============================")
            print("ATTENDANCE RECORDED")
            print("==============================")
            print("Student ID:", student_id_str)
            print("Student Name:", student_name)
            print("Emotion:", emotion)
            print(
                "Confidence:",
                f"{emotion_confidence:.2f}%"
            )
            print("Status:", status)
            print()

        # ==================================
        # DISPLAY
        # ==================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 255, 255),
            2
        )

        # Student information
        cv2.putText(
            frame,
            f"Student: {student_name}",
            (x, y - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ID: {student_id_str}",
            (x, y - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Emotion
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence: {emotion_confidence:.1f}%",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Status
        cv2.putText(
            frame,
            status,
            (x, y + h + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # ==================================
    # SHOW CAMERA
    # ==================================

    cv2.imshow(
        "Emotion Attendance System",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

camera.release()
cv2.destroyAllWindows()

print()
print("System closed.")