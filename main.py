import cv2

# Load pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access default camera
cap = cv2.VideoCapture(0)

# Check if camera is opened correctly
if not cap.isOpened():
    print("Unable to open video input.")
    exit()

while True:
    # Capture frame by frame from camera
    ret, frame = cap.read()

    # If frame wasn't captured, skip iteration
    if not ret:
        print("Unable to capture frame.")
        continue

    # Convert to gray scale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    # Draw rect around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame with detected faces
    cv2.imshow("Frame", frame)

    # Exit if user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break