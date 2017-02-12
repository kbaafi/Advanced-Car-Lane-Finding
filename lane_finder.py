import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
from image_thresholding import *
from lane_detection import *
import copy
from Pipeline import *
import sys

from moviepy.editor import VideoFileClip

if __name__=='__main__':
	arg_count =len(sys.argv)
	#print(arg_count)
	if arg_count!=3:
		print('Run this script as python lane_finder.py your_input_file.mp4 your_output_file.mp4')
	else:
		try:
			pipeline = Pipeline()			
			inputfile = str(sys.argv[1])
			outputfile = str(sys.argv[2])
			
			clip1 = VideoFileClip(inputfile)
			white_clip = clip1.fl_image(pipeline.process_image) #NOTE: this function expects color images!!
			white_clip.write_videofile(outputfile, audio=False)
		except Exception as e:
			print(str(e))
