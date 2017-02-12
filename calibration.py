import numpy as np
import glob
import cv2

def calibrate_camera(img_folder,name_pattern,chessboard_dims,resolution_dims):
	"""
	Given a set of chessboard images and the number of squares of the board in x and y directions,
	this function computes the camera matrix and the distortion coefficients
	
	Args:
		img_folder		: folder where chessboard images are stored
		name_pattern		: naming pattern of image files example 'calibration*,jpg'
		chessboard_dims		: dimensions of the chessboard (how many black squares in horizontal and vertival directions)
		resolution_dims		: pixel resolution of camera
	
	Returns:
		ret			: OpenCV return value
		dist_coeffs		: Distortion coefficients
		cam_matrix		: Camera Matrix
		rvecs			: rotational vectors
		tvecs			: translational vectors 
	"""
	objpts = []
	imgpts = []
	
	file_name_pattern = img_folder+'/'+name_pattern
	image_files = glob.glob(file_name_pattern)

	objpt = np.zeros((chessboard_dims[0]*chessboard_dims[1],3),np.float32)
	objpt[:,:2] = np.mgrid[0:chessboard_dims[0],0:chessboard_dims[1]].T.reshape(-1,2)

	for filename in image_files:
		img = cv2.imread(filename)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret,corners = cv2.findChessboardCorners(gray,chessboard_dims,None)
		
		if(ret==True):
			imgpts.append(corners)
			objpts.append(objpt)
		#end if
	#end for

	#calibration

	ret, cam_matrix,dist_coeffs,rvecs, tvecs = cv2.calibrateCamera(objpts,imgpts,resolution_dims,None,None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
	return ret,dist_coeffs,cam_matrix,rvecs,tvecs


def undistort(img,cam_matrix,dist_coeffs):
	"""
	Given an image the camera matrix and the distortion coefficients, this function computes the undistorted image
	
	Args:
		img			: original distorted image
		dist_coeffs		: Distortion coefficients
		cam_matrix		: Camera Matrix
	
	Returns:
		undistortedimage from cv2.undistort

	"""
	return cv2.undistort(img,cam_matrix,dist_coeffs,None,cam_matrix)


