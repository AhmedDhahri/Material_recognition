import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs2
import math
import time
import os
from threading import Thread, Lock
from collections import deque
import sys




frames_buffer = deque()
frames = None

pipeline = rs2.pipeline()
config = rs2.config()
pipeline_wrapper = rs2.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs2.stream.infrared, 1280, 720, framerate=6)
config.enable_stream(rs2.stream.depth, 1280, 720, framerate=6)
config.enable_stream(rs2.stream.color, 1280, 720, framerate=6)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()

#Set high density
depth_sensor.set_option(rs2.option.visual_preset, 4)

align = rs2.align(rs2.stream.color)

folder = 17
count_ts = 4350


count_img = 0
while True:
	frames = pipeline.wait_for_frames()
	aligned_frames = align.process(frames)

	nir_image = aligned_frames.get_infrared_frame()
	depth_frame = aligned_frames.get_depth_frame()
	color_frame = aligned_frames.get_color_frame()

	nir_image = np.asanyarray(nir_image.get_data())
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	nir_image = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2RGB)
	depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
	color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

	cv2.imshow('RGB', cv2.resize(nir_image, dsize=(640, 360)))
	k = cv2.waitKey(1)
	if k == ord('q'):
		break
	elif k == ord(' '):
		count_img = count_img + 1
		count_ts = count_ts + 1
		print(count_img, 'img_', count_ts)
		cv2.imwrite('files/img_raw/rgb/%06d'%count_ts + '_rgb.png', color_image)
		cv2.imwrite('files/img_raw/nir/%06d'%count_ts + '_nir.png', nir_image)
		cv2.imwrite('files/img_raw/dpt/%06d'%count_ts + '_dpt.png', depth_colormap)


cv2.destroyAllWindows()
