import numpy as np
from collections import deque

class Line():
	
	def __init__(self):
		# is this the first attempt
		self.first_attempt = True
		
		# number of times fitting has been tried
		self.n_attempts = 0
		
		# was the line detected in the last iteration?
		self.detected = False  

		# x values of the last n fits of the line
		self.recent_xfitted = []

		self.current_xfitted = []

		#average x values of the fitted line over the last n iterations
		self.bestx = None

		self.recent_fits = deque(maxlen = 13)    

		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  

		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([0,0,0])]

		#radius of curvature of the line in some units
		self.radius_of_curvature = 0

		#distance in meters of vehicle center from the line
		self.line_base_pos = None

		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')

		#x values for detected line pixels
		self.allx = None

		#y values for detected line pixels
		self.ally = None

		self.width = None

	def reset(self):
		self.n_attempts = 0
		self.detected = False
		
	def calc_radius(self,ym_per_pix,xm_per_pix,ploty):
		"""
		Calculates the radius of the line
		Args:
			xm_per_pix: X metres per pixel
			ym_per_pix: Y metres per pixel
		"""
		y_eval = np.max(ploty)
	
		# Fit new polynomials to x,y in world space
		fit_m = np.polyfit(ploty*ym_per_pix, self.current_xfitted*xm_per_pix, 2)

		# Calculate the new radii of curvature
		curve_rad = ((1 + (2*fit_m[0]*y_eval*ym_per_pix + fit_m[1])**2)**1.5) / np.absolute(2*fit_m[0])
		self.radius_of_curvature = curve_rad

	def calc_line_base_pos(self,xm_per_pix):
		"""
		Calculates the line's base position.
		Args:
			xm_per_pix: X metres per pixel
		"""
		base = self.current_xfitted[0]
		self.line_base_pos = xm_per_pix*base
		
	def fitx(self,fitcoeffs,x_pixels,y_pixels,ploty):
		"""
		Find the best fit for the current frame 
		Args:
			xm_per_pix: X metres per pixel
			ym_per_pix: Y metres per pixel
			ploty: sequence of elements
		"""
		self.allx = x_pixels
		self.ally = y_pixels

		if self.first_attempt == False:
			self.diffs = np.absolute(fitcoeffs-self.current_fit)
			if(len(x_pixels)<20):
				self.detected = False
				self.n_attempts+=1
			else:
				if(self.diffs[0]<=0.001):
					self.detected = True
					self.recent_fits.append(fitcoeffs)
					self.n_attempts = 0				
				else:
					self.detected = False
					self.n_attempts +=1
		else:
			if(len(x_pixels)<20):
				self.detected = False
				self.n_attempts+=1
			else:
				self.detected = True
				self.recent_fits.append(fitcoeffs)
				self.n_attempts = 0
			
			self.first_attempt = False
		
		
		self.current_fit = np.average(self.recent_fits, axis=0)	
		self.current_xfitted = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
		self.calc_radius((30/720),(3.7/700),ploty)
		self.calc_line_base_pos((3.7/700))

