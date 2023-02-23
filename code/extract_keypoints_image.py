import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.holistic

# Initialize MediaPipe Pose
pose = mp_pose.Holistic(static_image_mode=True, min_detection_confidence=0.75)

# Load the input image
image = cv2.imread('img.jpg')

# Resize image to (width=640, height=480)
image = cv2.resize(image, (450, 800))

# Convert the image to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract the pose landmarks
results = pose.process(image)

# Save the landmarks to a text file
#with open('landmarks.txt', 'w') as file:
#    for landmark in results.pose_landmarks.landmark:
#        file.write(f'{landmark.x} {landmark.y} {landmark.z}\n')

# Visualize the pose landmarks on the image
annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)

# Release resources
pose.close()
cv2.destroyAllWindows()
