from proj5_functions import *
from moviepy.editor import VideoFileClip
# white_output = 'test_videos_output/solidWhiteRight.mp4'
video_path = '../project_video.mp4'
output_video = '../output_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds

# clip1 = VideoFileClip(video_path).subclip(1,1)
clip1 = VideoFileClip(video_path)

# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output_video, audio=False)