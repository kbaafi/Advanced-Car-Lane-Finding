import numpy as np
import cv2


def get_yuv(img):
	"""
		Returns the YUV color space of an RGB image
	"""
	yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
	return yuv


def u_thresh(img,threshold):
	"""
		Image thresholding on U channel of YUV
		Args:
			img		: input image
			threshold	: masking threshold
		returns:
			u_binary	: binary image showing remaining pixels after transformation
	"""
	yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
	u_channel = yuv[:,:,1]

	u_scale_factor = np.max(u_channel)/255 
	u_scaled = (u_channel/u_scale_factor).astype(np.uint8)
        
	u_binary = np.zeros_like(u_scaled)
	u_binary[(u_scaled>=threshold[0])&(u_scaled<=threshold[0])]=1
	return u_binary

		
def hsv_range_thresh(img,hsv_low,hsv_high):
	"""
		Image thresholding on HSV image
		Args:
			img		: input image
			hsv_low 	: low hsv values
			hsv_high	: high hsv values
		returns:
			mask		: image after transformation
	"""
	hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	mask = cv2.inRange(hsv,hsv_low,hsv_high)
	return mask


def sobel_dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
		Performs direction sobel image filtering and thresholding
		Args:
			img		: input image
			sobel_kernel 	: sobel kernel size
			thresh		: high and low threshold values for filtering out unnecessary pixels
		returns:
			binary_output	: binary image showing remaining pixels after transformation
	"""
    
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# 2) Take the gradient in x and y separately
	sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
	sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
	# 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobel_x)
	abs_sobely = np.absolute(sobel_y)
	# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
	g_dir = np.arctan2(abs_sobely, abs_sobelx)
	# 5) Create a binary mask where direction thresholds are met
	binary_output = np.zeros_like(g_dir)
	binary_output[(g_dir>=thresh[0])&(g_dir<=thresh[1])]=1
	# 6) Return this mask as your binary_output image
	#binary_output = np.copy(img) # Remove this line
	return binary_output



def sobel_mag_thresh_gray(img,threshold,kernelsize):
	"""
		Performs sobel image filtering and thresholding on magnitude of filter in both x and y axes. The image is already presented as gray scale before transformation
		Args:
			img		: grayscale input image
			threshold	: high and low threshold values for filtering out unnecessary pixels
			kernel_size 	: sobel kernel size
		returns:
			binary_output	: binary image showing remaining pixels after transformation
	"""
	#gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = kernelsize)
	sobel_y = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = kernelsize)

	mag = np.sqrt(sobel_x**2+sobel_y**2)

	scale_factor = np.max(mag)/255 
	scaled = (mag/scale_factor).astype(np.uint8)

	binary_output = np.zeros_like(scaled)
	binary_output[(scaled>=threshold[0])&(scaled<=threshold[1])]=1
	return binary_output



def sobel_mag_thresh(img,threshold,kernelsize):
	"""
		Performs sobel image filtering and thresholding on magnitude of filter in both x and y axes.
		Args:
			img		: input image
			kernelsize 	: sobel kernel size
			threshold	: high and low threshold values for filtering out unnecessary pixels
		returns:
			binary_output	: binary image showing remaining pixels after transformation
	"""
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = kernelsize)
	sobel_y = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = kernelsize)

	mag = np.sqrt(sobel_x**2+sobel_y**2)

	scale_factor = np.max(mag)/255 
	scaled = (mag/scale_factor).astype(np.uint8)

	binary_output = np.zeros_like(scaled)
	binary_output[(scaled>=threshold[0])&(scaled<=threshold[1])]=1
	return binary_output


def abs_sobel_thresh(img,orient,threshold,kernelsize):
	"""
		Performs sobel image filtering and thresholding in one axes, x or y
		Args:
			img		: input image
			kernelsize 	: sobel kernel size
			threshold	: high and low threshold values for filtering out unnecessary pixels
		returns:
			binary_output	: binary image showing remaining pixels after transformation
	"""
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	if orient=='x':
		sobel_der = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = kernelsize)
	if orient == 'y':
		sobel_der = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = kernelsize)

	abs_sobel = np.absolute(sobel_der)
	scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
	binary_out = np.zeros_like(scaled)

	binary_out[(scaled>=threshold[0])&(scaled<=threshold[1])] = 1
	return binary_out


def hls_select(img,channel,threshold):
	"""
		Selects and performs thresholding on a channel in the HLS colorspace
		Args:
			img		: input image
			channel 	: selected HLS channel 's' or 'l' or 'h'
			threshold	: high and low threshold values for filtering out unnecessary pixels
		returns:
			binary_output	: binary image showing remaining pixels after transformation
	"""
	
	hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	if channel =='s':
		sel_channel = hsl[:,:,2]
	if channel == 'h':
		sel_channel = hsl[:,:,0]
	if channel == 'l':
		sel_channel = hsl[:,:,1]
	binary_out = np.zeros_like(sel_channel)
	binary_out[(sel_channel >= threshold[0]) & (sel_channel < threshold[1])] = 1
	return binary_out
