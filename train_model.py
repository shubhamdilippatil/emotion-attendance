import cv2
import os
import numpy as np

# ==============================
# SETTINGS
# ==============================

STUDENTS_FOLDER = "students"
MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "face_model.yml")

# Create model folder if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ==============================
# FACE RECOGNIZER
# ==============================

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haar Cascade
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []

# ==============================
# LOAD STUDENT DATA
# ==============================

if not os.path.exists(STUDENTS_FOLDER):
    print("ERROR: students folder does not exist.")
    exit()

student_folders = os.listdir(STUDENTS_FOLDER)

for student_id in student_folders:

    student_path = os.path.join(
        STUDENTS_FOLDER,
        student_id
    )

    # Ignore files
    if not os.path.isdir(student_path):
        continue

    # Student ID must be numeric
    if not student_id.isdigit():
        print("Skipping invalid student folder:", student_id)
        continue

    label = int(student_id)

    print("--------------------------------")
    print("Loading student:", student_id)

    image_count = 0

    for filename in os.listdir(student_path):

        image_path = os.path.join(
            student_path,
            filename
        )

        # Only process image files
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        # Read image in grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Could not read:", image_path)
            continue

        # Resize to same size
        image = cv2.resize(
            image,
            (200, 200)
        )

        faces.append(image)
        labels.append(label)

        image_count += 1

    print("Images:", image_count)

# ==============================
# CHECK DATA
# ==============================

if len(faces) == 0:
    print()
    print("ERROR: No face images found.")
    print("Please run register_student.py first.")
    exit()

print()
print("==============================")
print("FACE TRAINING")
print("==============================")

print("Total images:", len(faces))
print("Total labels:", len(labels))

# IMPORTANT:
# OpenCV LBPH requires labels to be a NumPy array
labels = np.array(
    labels,
    dtype=np.int32
)

# ==============================
# TRAIN
# ==============================

print()
print("Training face recognition model...")

recognizer.train(
    faces,
    labels
)

# ==============================
# SAVE MODEL
# ==============================

recognizer.write(
    MODEL_PATH
)

print()
print("==============================")
print("TRAINING COMPLETED")
print("==============================")

print("Model saved:", MODEL_PATH)