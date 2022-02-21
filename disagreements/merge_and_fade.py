import os
import subprocess
from os.path import join


def merge_and_fade(videos_dir, n_DAs, fade_out_frame, name, fade_duration=2, verbose=False,
                   opacity=0.99):
    """Creates bash file to merge the HL videos and add fade in fade out effects using ffmpeg"""

    """Create the necessary files"""
    f1 = open(join(videos_dir, "addFadeAndMerge.sh"), "w+")
    f1.write("#!/bin/bash\n")

    """merge with section before disagreement"""
    for i in range(n_DAs):
        f1.write(f"ffmpeg -f concat -safe 0 -i temp/together{i}.txt -c copy temp/merged{i}.mp4\n")

    """fade in/out"""
    for i in range(n_DAs):
        f1.write(f"ffmpeg -i temp/merged{i}.mp4 -filter:v "
                 f"'fade=in:{0}:{fade_duration},fade=out:{fade_out_frame}:{fade_duration}' "
                 f"-c:v libx264 -crf 22 -preset veryfast -c:a copy temp/fadeInOut_DA_{i}.mp4\n")

    """concatenate videos"""
    f1.write(f"ffmpeg -f concat -safe 0 -i temp/final_list.txt -c copy {name}_DA.mp4")
    f1.close()

    """create files of videos to concatenate"""
    f2 = open(join(videos_dir, "temp/final_list.txt"), "w+")
    for i in range(n_DAs):
        f2.write(f"file fadeInOut_DA_{i}.mp4\n")
    f2.close()

    """create files of videos to concatenate"""
    for i in range(n_DAs):
        f = open(join(videos_dir, f"temp/together{i}.txt"), "w+")
        f.write(f"file together{i}.mp4\n")
        f.write(f"file a1_DA{i}.mp4\n")
        f.close()


    """make executable"""
    current_dir = os.getcwd()
    os.chdir(videos_dir)
    subprocess.call(["chmod", "+x", "addFadeAndMerge.sh"])
    """call ffmpeg"""
    if verbose:
        subprocess.call("./addFadeAndMerge.sh")
    else:
        subprocess.call("./addFadeAndMerge.sh", stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
    os.chdir(current_dir)  # return
