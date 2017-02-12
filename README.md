
**Advanced Lane Finding Project**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

##Usage
```
python lane_finder.py inputfile outputfile
	--inputfile	:File name of input video
	--outputfile	:File name of output video 
```
---


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./rubric_images/original_cam_input.png "Original Camera Capture"
[image2]: ./rubric_images/undistorted.png "Undistorted Image"
[image3]: ./rubric_images/warped.png "Warped Image"
[image4]: ./rubric_images/sobel_on_l.png "Sobel on HLS(L)"
[image5]: ./rubric_images/sobel_on_s.png "Sobel on HLS(L)"
[image6]: ./rubric_images/white_mask.png "White Mask"
[image7]: ./rubric_images/yellow_mask.png "Yellow Mask"
[image8]: ./rubric_images/combined.png "Combined Thresholding"
[image9]: ./rubric_images/final.png "Final Output"
[image10]: ./rubric_images/hist.png "Histogram"
[image11]: ./rubric_images/fitted.png "Polynomial Fitted Lines"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

####README

###Camera Calibration
The code for this step is contained in the file `calibration.py`. The file contains functions for calibrating the camera and performing a transformation of captured images to their undistorted representations.

Calibration of the camera is done by using a series of images of chessboards. Our list of images is found in the folder `camera_cal`. We maintain a list of object points `objpts` which represent the physical location of the corners of the chessboard in each image in the folder. We also maintain the coordinates of locations where the black squares form a corner on the 9x6 board. The OpenCV function `cv2.findChessboardCorners` assigns each corner to its position in the image. We maintain the list of corners and physical points and use this to calibrate the camera.

The function `cv2.calibrateCamera` returns among others the distortion coefficients and the camera matrix. The captured camera image can then be calibrated using `cv2.undistort`

The result of undistorting a captured image is shown here

![Distorted image][image1]

![Undistorted image][image2]

###Pipeline (single images)
The pipeline processes one image at a time and this is fed in by the `moviepy.editor` `VideoFileClip.fl_image` function. see line 21 to 26 of `lane_finder.py`

Before running the pipeline the camera is calibrated to obtain the camera matrix and the distortion coefficients

The steps performed by the pipeline are as follows:

####1. Distortion Correction
An example undistorted image is shown below.:

![alt text][image2]

####2. Perspective Transformation
The code for the transformation is located in lines 33 to 39 of the `Pipeline.process_image` function. The source and destination points for the were hardcoded but were chosen such that the base and top lines of our region of interest will have the same length in our transformed image. The source and destination points are shown below:

```
		src_vertices = np.array([(240,imshape[0]),(580, 450), (720, 450), (1200,imshape[0])],np.float32)
		dst_vertices = np.array([(360,imshape[0]),(240,0), (1200,0), (1080,imshape[0])],np.float32)

```
This resulted in the following source and destination points: 
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 240, 720      | 360, 720      | 
| 580, 450      | 240, 0        |
| 720, 450      | 1200, 0       |
| 1200, 720     | 1080, 720     |

As you can see below, after the perspective transform we obtain a result such that both lines are parallel and not converging to a horizon.

Original Image

![Original image][image2]

Image after perspective transform

![Image after perspective transform][image3]

####3. Color transforms to reveal the lines
The following combination of color and gradient thresholds were used to isolate the lines. The functions that perform these tasks are in the file `image_thresholding.py`  and are called in lines 43 to 58 of `Pipeline.py`
| Method        | Threshold or Range   | 
|:-------------:|:-------------:| 
| Magnitude of Sobel in x and y directions performed on L channel of HLS      | 50 - 200     | 
| Magnitude of Sobel in x and y directions performed on S channel of HLS       | 50 - 200       |
| Range masking of HSV     | between  [0, 100, 100] and  [50,255,255]    |
| Range masking of HSV    | between [20, 0, 180] and [255,80,255]     |

The effects of the various transforms are shown below together with their combined effect is shown below

Sobel Magnitude Theshold on L Channel in HLS

![alt text][image4]

Sobel Magnitude Theshold on S Channel in HLS

![alt text][image5]

White Masking

![alt text][image6]

Yellow Masking

![alt text][image7]

Combined Thresholding and Masking

![alt text][image8]

####4. Identifying the lines with good confidence
The code for this section is implemented in `lane_detection.py`

We begin by focusing on the lower half of the transformed image. We take a histogram of the image. Since the street is asphalt colored with white or yellow markings, we expect to find the left marking at the peak position of the histograms. This is implemented in the function `detect_lanes_with_histogram`. This gives us a starting position for searching for line points since nonzero points in the transformed image signal the possibility of a line. We add a margin to y location the left and right histogram peaks and use a sliding window to successively find the lane lines moving from the bottom to the top. The number of windows is determined by the `nWindow` variable. If we find more than a minimum value of pixels within a search location, we expect more nonzero pixels in that vicinity so we reposition our search window to the mean of the the number of pixels obtained and use this as the search area in the next window.

When the pixel coordinates `(x and y)` of the lanes are found, we then fit a 2nd degree polynomial to obtain the line. With this we can produce a line function that predicts where the lane is. 

Searching by using the histogram is done at the beginning of the video. If we are able to fit a line, in the next frame we look for line pixels in the vicinity of a margin of the fitted line from the current frame. The function that performs this task is called `detect_lanes_without_histogram`

Using the histogram to locate the peaks of pixel densities

![alt text][image10]

##### Line search heuristics
The following heuristics were used: see lines 61 to 78 in `Pipeline.py` and the `fitx` function in `Line.py`
`
	At start of program search for the lines using histograms

	Maintain a list of previous fits

	If the line pixels were found, search for lines in vicinity of the obtained lines and add the current fit to the list of fits

	If the number of pixels is less than a threshold we assume that line was not found
		The next line is deduced from the average of previously  obtained fits

	If the quadratic coefficient of the current fit is greater than 0.001 we assume that line was not found
		The next line is deduced from the average of previously  obtained fits

	Ultimately: deduce the fit from the average of the fits obtained

	After 3 failures of finding any line, use the histogram method, but search within an outer margin of the previously obtained lines (see detect_lanes_with_histogram_and_bounds in lane_detection)
`
Polynomial Fitting example

![alt text][image11]

####5. Radius of the curvature and the distance from center.
Radius of curvature is implemented in the `calc_radius` function in the class `Line.py` it takes in pixel-to-meter constants for the x and y coordinates and calculates the radius of the fitted line in meters

The base position of each line is found by taking the its base line value. The difference between these points is the position of the car in the lanes. The distance from the center of the image is the distance from the center of the image to this difference. The base position is calculated in the function `calc_line_base_pos` in `Line.py`. Lines 94 to 96 in `Pipeline.py` calculate this offset

#### An example result.
The lines are drawn by the function `draw_detected_lane` in `lane_detection.py`. An example result is shown below:

![alt text][image9]

---

### Resulting Video

Here's a [link to the processed video](https://www.youtube.com/watch?v=-BephdgM3g4&feature=youtu.be)

Also here's the [pipeline's output on the challenge_video.mp4](https://www.youtube.com/watch?v=dX2aiVbUfe8)

---

###Discussion
The choice of image processing functions for determining the the lane pixels were the most tricky parts of the project. Several options and tweaks make this a very laborious process. The variation of ambient lighting and coloration of the road itself also makes the search for the lanes very difficult.

The next problem is how to determine an averaging scheme or recover from a series of misses on the road. On testing the challenge video, lighting conditions cause the lane to veer off. But because of the averaging method used, it takes a while before the correct adjustment is effected. 


