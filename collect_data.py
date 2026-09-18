import cv2
import os

emotion = input("Enter emotion (happy / neutral / sad): ").lower()

if emotion not in ["happy", "neutral", "sad"]:
    print("Please enter happy, neutral or sad.")
    exit()

folder = os.path.join("dataset", emotion)

os.makedirs(folder, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print("Camera started.")
print("Show your", emotion, "expression.")
print("Press Q to stop.")

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for x, y, w, h in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (48, 48)
        )

        filename = os.path.join(
            folder,
            f"{emotion}_{count}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        count += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{emotion}: {count}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Collect Training Data",
        frame
    )

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Data collection completed.")
print("Images collected:", count)