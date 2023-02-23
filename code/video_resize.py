import cv2

# open the video
video = cv2.VideoCapture('vid_test.mp4')

# get the video properties
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# set the new dimensions
new_width = 1280
new_height = 720

# create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (new_width, new_height))

# read and resize the frames
for i in range(frame_count):
    ret, frame = video.read()
    if ret:
        resized_frame = cv2.resize(frame, (new_width, new_height))
        output.write(resized_frame)

# release the video objects
video.release()
output.release()
cv2.destroyAllWindows()
