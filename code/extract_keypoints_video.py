import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture('vid_test.mp4')
file = open('keypoints.txt', 'w')


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB format and process it with Mediapipe
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # Extract the pose landmarks
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            file.write(f'{landmark.x} {landmark.y} {landmark.z}\n')

    # Display the processed image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
