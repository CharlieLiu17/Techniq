from moviepy.video.io.VideoFileClip import VideoFileClip

input_video_path = './test_inputs/video/frisbee/brodie_far_tutorial.mp4'
output_video_path = './test_inputs/video/frisbee/brodie_flick2.mp4'

with VideoFileClip(input_video_path) as video:
    new = video.subclip(253, 256)
    new.write_videofile(output_video_path, audio_codec='aac')