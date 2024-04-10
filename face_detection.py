import cv2
import time

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize variables
face_detected = False
start_time = None
total_detection_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Update face detection status and start time (only on first detection)
    if len(faces) > 0 and not face_detected:
        face_detected = True
        start_time = time.time()

    # Stop timer and calculate total time when face is no longer detected
    elif len(faces) == 0 and face_detected:
        face_detected = False
        end_time = time.time()
        total_detection_time += end_time - start_time
        start_time = None  # Reset start time for next detection

    # Draw rectangle around detected faces (optional)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display total face detection time (if a face is currently detected)
    if face_detected:
        current_time = time.time()
        elapsed_time = current_time - start_time
        text = f"Total Face Detection Time: {total_detection_time + elapsed_time:.2f} seconds"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame with bounding boxes and time (optional)
    cv2.imshow('Face Detection with Time', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()
