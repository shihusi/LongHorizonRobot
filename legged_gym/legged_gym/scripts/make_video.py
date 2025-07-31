import cv2
import os
from tqdm import tqdm

env_name = "elevator_replan"

image_folder = f'./logs/llm_{env_name}/exported/frames'  # directory to the pictures
video_name = f'./llm_{env_name}_v5.mp4'  # name for the video

print("image folder=", image_folder)
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = sorted(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0x7634706d, 50, (width,height))

for image in tqdm(images):
    # print(image)
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

cv2.destroyAllWindows()
video.release()