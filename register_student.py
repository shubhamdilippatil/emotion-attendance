import cv2
import os
import json


# ==========================================
# SETTINGS
# ==========================================

STUDENTS_FOLDER = "students"
MODEL_FOLDER = "model"
STUDENT_FILE = os.path.join(
    MODEL_FOLDER,
    "students.json"
)


# ==========================================
# CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs(
    STUDENTS_FOLDER,
    exist_ok=True
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ==========================================
# GET STUDENT INFORMATION
# ==========================================

student_id = input(
    "Enter Student ID: "
).strip()

student_name = input(
    "Enter Student Name: "
).strip()


if not student_id:

    print("Student ID cannot be empty.")
    exit()


if not student_name:

    print("Student Name cannot be empty.")
    exit()


# ==========================================
# CHECK STUDENT ID
# ==========================================

if not student_id.isdigit():

    print("Student ID must contain numbers only.")

    exit()


# ==========================================
# CREATE STUDENT FOLDER
# ==========================================

student_folder = os.path.join(
    STUDENTS_FOLDER,
    student_id
)

os.makedirs(
    student_folder,
    exist_ok=True
)


# ==========================================
# LOAD STUDENT DATABASE
# ==========================================

if os.path.exists(STUDENT_FILE):

    with open(
        STUDENT_FILE,
        "r"
    ) as file:

        students = json.load(file)

else:

    students = {}


# ==========================================
# SAVE STUDENT INFORMATION
# ==========================================

students[student_id] = student_name


with open(
    STUDENT_FILE,
    "w"
) as file:

    json.dump(
        students,
        file,
        indent=4
    )


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    print("ERROR: Face detector failed.")

    exit()


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ==========================================
# COUNT EXISTING IMAGES
# ==========================================

existing_images = [

    file for file in os.listdir(
        student_folder
    )

    if file.lower().endswith(".jpg")
]


count = len(existing_images)


print()
print("==============================")
print("STUDENT REGISTRATION")
print("==============================")
print("ID:", student_id)
print("Name:", student_name)
print()
print("Look at the camera.")
print("Move your head slowly.")
print("Press Q to stop.")
print()


# ==========================================
# COLLECT FACE DATA
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Camera error.")

        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    for x, y, w, h in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]


        face = cv2.resize(
            face,
            (200, 200)
        )


        filename = os.path.join(
            student_folder,
            f"face_{count}.jpg"
        )


        cv2.imwrite(
            filename,
            face
        )


        count += 1


        # Face rectangle

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Images: {count}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    cv2.putText(
        frame,
        f"Student: {student_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to stop",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Student Registration",
        frame
    )


    if cv2.waitKey(100) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()


print()
print("==============================")
print("REGISTRATION COMPLETED")
print("==============================")
print("Student ID:", student_id)
print("Student Name:", student_name)
print("Images:", count)
print(
    "Folder:",
    student_folder
)