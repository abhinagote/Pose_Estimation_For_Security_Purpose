import cv2
import mediapipe as mp
import winsound
import time

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the video stream from the camera
cap = cv2.VideoCapture(0)
start_time = None

while True:
    # Read each frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the color of the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the MediaPipe Pose model to detect the pose in the current frame
    results = pose.process(frame)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2,
                                                                         circle_radius=2),
                               connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                           circle_radius=2))

        # Check if the hands-up pose is detected
        hands_up = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y < results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.LEFT_SHOULDER].y and \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y < \
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if hands_up:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 5:
                print("Hands up for more than 10 seconds!")
                winsound.Beep(770, 455)
        else:
            start_time = None

    # Convert the frame color back to BGR for displaying
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show the frame with the pose landmarks drawn on it
    cv2.imshow("Pose Estimation", frame)

    # Wait for the 'q' key to be pressed to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()