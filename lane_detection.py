import numpy as np
import cv2
from image_thresholding import *


def draw_message_box(img,topline,bottomline):
	"""
	Draws the statistics of the current frame on the image
	Args:
		img		: the image
		topline		: the first line
		bottomline	: the bottom line
	Returns:
		result		: the resulting image
	"""
	img_zero = np.zeros_like(img).astype(np.uint8)
	recpoints = (100,480)
	rect = cv2.rectangle(img_zero,(50,50),(50+recpoints[1],50+recpoints[0]),[0,255,0],thickness = -1)
	result = cv2.addWeighted(img, 1, rect, 0.3, 0)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result,topline,(75,75), font, 0.7,(51,51,51),2)
	cv2.putText(result,bottomline,(75,120), font, 0.7,(51,51,51),2)
	
	return result


def draw_detected_lane(undist,warped,Minv,left_fitx,right_fitx,ploty):
	"""
	Draws the detected lane on the undistorted image
	Args:
		undist		: the undistorted image
		warped		: the warped version of the image
		Minv		: the inverse perspective transform
		left_fitx	: the fitted left line
		right_fitx	: the fitted right line
		ploty		: line space for the fitted curve
	Returns:
		result		: the resulting image
	"""	

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	return result


def detect_lanes_without_histogram(binary_warped,margin,curr_left_fit,curr_right_fit):
	"""
	Detects the lanes by using previously fitted liness
	Args:
		binary_warped	: the binary image
		margin		: the margin around the fitted line
		curr_left_fit	: the currently fitted line on the left
		curr_right_fit	: the currently fitted line on the right
	Returns:
		left_fit	: the coefficients of the fitted line on the left
		right_fit	: the coefficients of the fitted line on the right
		leftx		: the x coordinate of the left line
		lefty		: the y coordinate of the left line
		rightx		: the x coordinate of the right line
		righty		: the y coordinate of the left line
	"""		
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	#margin = 50

	left_lane_inds = ((nonzerox > (curr_left_fit[0]*(nonzeroy**2) + curr_left_fit[1]*nonzeroy + curr_left_fit[2] - margin)) & (nonzerox < (curr_left_fit[0]*(nonzeroy**2) + curr_left_fit[1]*nonzeroy + curr_left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (curr_right_fit[0]*(nonzeroy**2) + curr_right_fit[1]*nonzeroy + curr_right_fit[2] - margin)) & (nonzerox < (curr_right_fit[0]*(nonzeroy**2) + curr_right_fit[1]*nonzeroy + curr_right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	return left_fit,right_fit,leftx,lefty,rightx,righty


def detect_lanes_with_histogram_and_bounds(binary_warped_img,margin, bounds):
	"""
	Detects the lanes by using previously fitted liness
	Args:
		binary_warped	: the binary image
		margin		: the margin around bounds
		bounds		: the search boundary
	Returns:
		left_fit	: the coefficients of the fitted line on the left
		right_fit	: the coefficients of the fitted line on the right
		leftx		: the x coordinate of the left line
		lefty		: the y coordinate of the left line
		rightx		: the x coordinate of the right line
		righty		: the y coordinate of the left line
	"""
	
	adjusted_warped = binary_warped_img
	#adjusted_warped[:,0:bounds[0]-(20)] = 0
	adjusted_warped[:,bounds[1]+(4*margin):binary_warped_img.shape[1]] = 0

	histogram = np.sum(binary_warped_img[binary_warped_img.shape[0]/2:,:], axis=0)
	
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint-50])
	rightx_base = np.argmax(histogram[midpoint+50:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped_img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	#margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 25
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped_img.shape[0] - (window+1)*window_height
		win_y_high = binary_warped_img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		#print("right",good_right_inds.shape)
		
		# Append these indices to the lists
		left_lane_inds = np.append(left_lane_inds,[good_left_inds])
		right_lane_inds = np.append(right_lane_inds,[good_right_inds])
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = left_lane_inds.astype('int')
	right_lane_inds = right_lane_inds.astype('int')
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	return left_fit,right_fit,leftx,lefty,rightx,righty



def detect_lanes_with_histogram(binary_warped_img,margin):
	"""
	Detects the lanes by using a sliding window on the image to detect the line 
	pixels after using the peaks of pixel densities
	Args:
		binary_warped	: the binary image
		margin		: the margin around 
	Returns:
		left_fit	: the coefficients of the fitted line on the left
		right_fit	: the coefficients of the fitted line on the right
		leftx		: the x coordinate of the left line
		lefty		: the y coordinate of the left line
		rightx		: the x coordinate of the right line
		righty		: the y coordinate of the left line
	"""

	histogram = np.sum(binary_warped_img[binary_warped_img.shape[0]/2:,:], axis=0)
	
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint-50])
	rightx_base = np.argmax(histogram[midpoint+50:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped_img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	#margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 25
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped_img.shape[0] - (window+1)*window_height
		win_y_high = binary_warped_img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		
		# Append these indices to the lists
		left_lane_inds = np.append(left_lane_inds,[good_left_inds])
		right_lane_inds = np.append(right_lane_inds,[good_right_inds])
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = left_lane_inds.astype('int')
	right_lane_inds = right_lane_inds.astype('int')
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	return left_fit,right_fit,leftx,lefty,rightx,righty



def detect_lanes_with_histogram_hist(binary_warped_img,margin):
	"""
	Detects the lanes by using a sliding window on the image to detect the line 
	pixels after using the peaks of pixel densities
	Args:
		binary_warped	: the binary image
		margin		: the margin around 
	Returns:
		left_fit	: the coefficients of the fitted line on the left
		right_fit	: the coefficients of the fitted line on the right
		leftx		: the x coordinate of the left line
		lefty		: the y coordinate of the left line
		rightx		: the x coordinate of the right line
		righty		: the y coordinate of the left line
		histogram	: the histogram of the lower half of binary_warped
	"""

	histogram = np.sum(binary_warped_img[binary_warped_img.shape[0]/2:,:], axis=0)
	
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint-50])
	rightx_base = np.argmax(histogram[midpoint+50:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped_img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	#margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 25
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped_img.shape[0] - (window+1)*window_height
		win_y_high = binary_warped_img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		
		# Append these indices to the lists
		left_lane_inds = np.append(left_lane_inds,[good_left_inds])
		right_lane_inds = np.append(right_lane_inds,[good_right_inds])
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = left_lane_inds.astype('int')
	right_lane_inds = right_lane_inds.astype('int')


	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	return left_fit,right_fit,leftx,lefty,rightx,righty,histogram
