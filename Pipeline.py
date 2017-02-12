from lane_detection import *
from image_thresholding import *
from calibration import *
from Line import *

import cv2


class Pipeline():
	"""
	Processes each video image, draws the deduced lanes 
	and calculates the radius of curvature and the car's offset
	from the center
	"""

	left_fit = []
	right_fit = []

	# camera matrix
	c_matrix = []

	# distortion coefficients
	dist_coeffs = []

	# the left line
	left_line = Line()

	# the right line
	right_line = Line()

	# flag that decides whether to search for the lines using the 
	# histogram method
	search_by_histogram = True

	def __init__(self):
		self.reset()

	def reset(self):
		self.left_fit = []
		self.right_fit = []
		ret,dist_coeffs,cam_matrix,rvecs,tvecs = calibrate_camera('camera_cal','calibration*.jpg',(9,6),(720,1280))
		self.c_matrix = cam_matrix
		self.dist_coeffs = dist_coeffs
		self.left_line = Line()
		self.right_line = Line()
		self.search_by_histogram = True
	
	
	def process_image(self,img):
		"""
		Detects the lanes on the road and overlays the deduced lane on the
		road
		"""
		# image undistortion
		ret_img = np.array(undistort(img,self.c_matrix,self.dist_coeffs))
		imshape = img.shape
		
		# search margin
		margin = 50

		# source and destination vertices for bird's eye view perspective transform
		src_vertices = np.array([(240,imshape[0]),(580, 450), (720, 450), (1200,imshape[0])],np.float32)
		dst_vertices = np.array([(360,imshape[0]),(240,0), (1200,0), (1080,imshape[0])],np.float32)
		
		# perspective matrixes
		M = cv2.getPerspectiveTransform(src_vertices,dst_vertices)
		Minv = cv2.getPerspectiveTransform(dst_vertices,src_vertices)

		# the warped image
		warped = cv2.warpPerspective(ret_img, M, (1280,720), flags=cv2.INTER_LINEAR)

		# HLS representation
		warped_hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)

		# Sobel thresholding
		sx_image = sobel_mag_thresh_gray(warped_hls[:,:,1],(50,200),3)
		sx_image2 = sobel_mag_thresh_gray(warped_hls[:,:,2],(50,200),3)

		# Yellow HSV mask
		hsv_low = np.array([ 0, 100, 100])
		hsv_high = np.array([ 50, 255, 255])
		ymask_image = hsv_range_thresh(warped,hsv_low,hsv_high)

		# White HSV mask
		hsv_low = np.array([  20,   0,   180])
		hsv_high = np.array([ 255,  80, 255])
		wmask_image = hsv_range_thresh(warped,hsv_low,hsv_high)

		# Combined masks and filters
		total = np.logical_or(sx_image,wmask_image).astype('float32')
		total = np.logical_or(total,ymask_image).astype('float32')
		total = np.logical_or(total,sx_image2).astype('float32')

		# detecting the points of the lane lines
		if(self.search_by_histogram==True):
			if(len(self.left_line.current_xfitted)==0 or len(self.right_line.current_xfitted)==0):
				left_fit,right_fit,leftx,lefty,rightx,righty = detect_lanes_with_histogram(total,margin)
			else:
				minpt = int(np.amin(self.right_line.current_xfitted))
				maxpt = int(np.amin(self.right_line.current_xfitted))
				
				left_fit,right_fit,leftx,lefty,rightx,righty = detect_lanes_with_histogram_and_bounds(total,margin,(minpt,maxpt))
		else:
			left_fit,right_fit,leftx,lefty,rightx,righty = detect_lanes_without_histogram(total,margin,self.left_fit,self.right_fit)
		
		# decide what next search method will be
		if(self.left_line.n_attempts == 5 or self.right_line.n_attempts ==5):
			self.search_by_histogram = True
			self.left_line.reset()
			self.right_line.reset()
		else:
			self.search_by_histogram = False

		self.left_fit = left_fit
		self.right_fit  = right_fit

		ploty = np.linspace(400, warped.shape[0]-1, warped.shape[0]-400 )

		# deduce left line
		self.left_line.fitx(left_fit,leftx,lefty,ploty)

		# deduce right line
		self.right_line.fitx(right_fit,rightx,righty,ploty)

		res_img = draw_detected_lane(ret_img,total,Minv,self.left_line.current_xfitted,self.right_line.current_xfitted,ploty)

		# offset from lane center
		offset_from_center = self.right_line.line_base_pos - self.left_line.line_base_pos
		center = imshape[1]/2*(3.7/700)
		distance_from_center = np.absolute(center-offset_from_center)

		ltext = 'Curve Radius: '+' '+str(self.left_line.radius_of_curvature)
		rtext = 'Offset from Center: '+' '+str(distance_from_center)

		res_img = draw_message_box(res_img,ltext,rtext)
		
		return res_img
		
