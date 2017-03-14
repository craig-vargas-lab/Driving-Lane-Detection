import numpy as np
import glob
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv_utils

from moviepy.editor import VideoFileClip

# Constants
# =========
IN_VID = 'project_video.mp4'
OUT_VID = 'tracked.mp4'

TEST_DIR = 'test_images' # 'more_test_images' # 'test_images'
GLOB_EXT = 'cv_test*.jpg' # 'test*.jpg'
TEST_NAME = 'test1.jpg'
STRAIGHT = 'straight_lines1.jpg'
TEST_IMG = 'straight_lines1.jpg'
OUT_DIR = 'output_images' # 'cvOutput' # 'output_images'
OUT_NAME_PREFIX = 'processed'

# Run Mode Constants
# ==================
DISPLAY_IMAGE = True
PROCESS_TEST_IMAGES = False
PROCESS_VIDEO = False
# Pipeline options
# ==================
SAVE_PIPELINE = True
DISPLAY_PIPELINE = False

# Calibration data
# ================
with open('calibration.p', 'rb') as pickle_in:
	dist_pickle = pickle.load(pickle_in)
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def main():
	if PROCESS_VIDEO:
		process_video()
	else:
		if DISPLAY_IMAGE:
			test_pipeline()
		elif PROCESS_TEST_IMAGES:
			process_test_images()
		else:
			print("No run mode selected")


def process_video():
	"""
	Function processes an input video specified in the constants section of the code above.
	"""
	clip1 = VideoFileClip(IN_VID)
	video_clip = clip1.fl_image(process_img)
	video_clip.write_videofile(OUT_VID, audio=False)


def test_pipeline():
	"""
	Function was used for testing purposes to test the image processing pipeline
	"""
	img = mpimg.imread(os.path.join(TEST_DIR, TEST_IMG))
	if DISPLAY_IMAGE:
		# Show image
		plt.imshow(img)
		plt.show()
	display = process_img(img)


def process_test_images():
	"""
	Function loops through all test images and runs them through the image processing pipeline
	to find lanes
	"""
	image_paths = glob.glob(os.path.join(TEST_DIR, GLOB_EXT))
	for idx, path in enumerate(image_paths):
		print('Currently working on:', path)
		img = mpimg.imread(path)
		display = process_img(img)
		cv2.imwrite(os.path.join(OUT_DIR, OUT_NAME_PREFIX + str(idx + 1) + ".jpg"), cv2.cvtColor(display, cv2.COLOR_RGB2BGR))


def process_img(img):
	"""
	Function takes in an image and process it from end to end to output the lines and data.
	"""
	undistorted = undistort_img(img)
	binary = get_binary_img(undistorted)
	birds_eye, M, Minv = get_birds_eye_view(binary)
	color_mask_warp, left_curverad, right_curverad, car_offset_meters = find_lanes(birds_eye)
	display = get_display_img(img, color_mask_warp, Minv, left_curverad, right_curverad, car_offset_meters, display_image=DISPLAY_IMAGE)

	return display


def get_display_img(img, color_mask_warp, Minv, left_curverad, right_curverad, car_offset_meters, display_image=True):
	"""
	Function performs the final processing of the pipeline where the information on position and curvature
	is displayed on the final image as well as the highlighted lanes that were found
	"""
	# Set up display text
	curvature_text = 'Curvature at mid-screen height -> Left = {0:.3f},  Right = {1:.3f}'.format(left_curverad, right_curverad)
	if car_offset_meters < 0:
		offset_text = 'Car is {0:.3f} meters left of center'.format(-1*car_offset_meters)
	else:
		offset_text = 'Car is {0:.3f} meters right of center'.format(car_offset_meters)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	color_mask = cv2.warpPerspective(color_mask_warp, Minv, (img.shape[1], img.shape[0])) 

	# Combine the result with the original image
	display = cv2.addWeighted(img, 1, color_mask, 0.3, 0)
	cv2.putText(display, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
	cv2.putText(display, offset_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

	if display_image:
		# Show image
		plt.imshow(display)
		plt.show()

	if SAVE_PIPELINE:
		out_name = 'final_display_test4.jpg'
		out_path = os.path.join(OUT_DIR, out_name)
		cv2.imwrite(out_path, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

	return display





def get_binary_img(img):
	"""
	Function creates a binary version of the imput image that removes most information except for the lanes
	that the car must stay within.
	"""



	# # # Filtering for S channel then taking sobel of that image to clean it up
	# hls_s = cv_utils.hls_select(img, channel='s', thresh=(170,255)) # ***Helpful (170,255)
	# # hsv_s = cv_utils.hsv_select(img, channel='s', thresh=(205,255)) # ***Helpful (125,255)



	# hls_s1 = cv_utils.hls_select(img, channel='s', thresh=(220,255)) # ***Helpful (100,255)
	# hls_s2 = cv_utils.hls_select(img, channel='s', thresh=(230,255)) # ***Helpful (100,255)
	# hls_s3 = cv_utils.hls_select(img, channel='s', thresh=(240,255)) # ***Helpful (100,255)

	# hsv_v1 = cv_utils.hsv_select(img, channel='v', thresh=(225,255)) # ***Helpful (100,255)
	# hsv_v2 = cv_utils.hsv_select(img, channel='v', thresh=(227,255)) # ***Helpful (100,255)
	# hsv_v3 = cv_utils.hsv_select(img, channel='v', thresh=(239,255)) # ***Helpful (100,255)



	# # kernal test
	# sobelx = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(19, 255))
	# sobely = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(19, 255)) # ***Good

	# sobelx7 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(24, 255)) # Winner
	# sobelx8 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(27, 255))
	# sobelx9 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(30, 255))
	# sobely = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(5, 255)) 

	# sobelx2 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(15, 255))
	# sobely2 = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(40, 255))
	# sobel2 = np.zeros_like(hls_s)
	# sobel2[((sobelx2==1) & (sobely2==1))] = 1

	# sobelx3 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(18, 255))
	# sobely3 = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(50, 255))
	# sobel3 = np.zeros_like(hls_s)
	# sobel3[((sobelx3==1) & (sobely3==1))] = 1 

	# sobelx4 = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(21, 255))
	# sobely4 = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(65, 255))
	# sobel4 = np.zeros_like(hls_s)
	# sobel4[((sobelx4==1) & (sobely4==1))] = 1 


	# binary = np.zeros_like(hls_s)
	# # binary[(hls_s == 1) | (hsv_s == 1) | ((sobelx==1) & (sobely==1))] = 1 # Winner
	# binary[(hls_s == 1) | ((sobelx==1) & (sobely==1))] = 1 # Winner


	# lab_l = cv_utils.lab_select(img, channel='l', thresh=(200,255))
	# lab_a = cv_utils.lab_select(img, channel='a', thresh=(135,255))
	# lab_b = cv_utils.lab_select(img, channel='b', thresh=(135,255))
	# lab_b2 = cv_utils.lab_select(img, channel='b', thresh=(140,255))
	# lab_b3 = cv_utils.lab_select(img, channel='b', thresh=(145,255)) # Winner


	# luv_l = cv_utils.luv_select(img, channel='l', thresh=(212,255))
	# luv_l2 = cv_utils.luv_select(img, channel='l', thresh=(214,255))
	# luv_l3 = cv_utils.luv_select(img, channel='l', thresh=(216,255)) # Winner
	# luv_u = cv_utils.luv_select(img, channel='u', thresh=(105,255))
	# luv_v = cv_utils.luv_select(img, channel='v', thresh=(150,255))

	# bin_test = np.zeros_like(hls_s)
	# bin_test[(lab_b==1) | (luv_l3==1) | ((sobelx==1) & (sobely==1))] = 1
	# sobel = np.zeros_like(hls_s)
	# sobel[((sobelx==1) & (sobely==1))] = 1


	# # Visualize
	# sobel_comb = np.zeros_like(sobelx)
	# sobel_comb[(sobelx==1) & (sobely==1)] = 1
	# f, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(3, 3, figsize=(9,6))
	# ax1.set_title('Sobel2')
	# ax1.imshow(hls_s1, cmap='gray')
	# ax2.set_title('Sobel3')
	# ax2.imshow(hls_s2, cmap='gray')
	# ax3.set_title('Sobel4')
	# ax3.imshow(hls_s3, cmap='gray')
	# # ==============================
	# ax4.set_title('SobelX2')
	# ax4.imshow(lab_b, cmap='gray')
	# ax5.set_title('SobelX3')
	# ax5.imshow(lab_b2, cmap='gray')
	# ax6.set_title('SobelX4')
	# ax6.imshow(lab_b3, cmap='gray')
	# # ==============================
	# ax7.set_title('Test')
	# ax7.imshow(hsv_v1, cmap='gray')
	# ax8.set_title('Old')
	# ax8.imshow(hsv_v2, cmap='gray')
	# ax9.set_title('Sobel')
	# ax9.imshow(hsv_v3, cmap='gray')
	# plt.show()
	# # exit()
	# # Visualize part II
	# # Binary filters
	# lab_b = cv_utils.lab_select(img, channel='b', thresh=(135,255)) # Winner
	# luv_l = cv_utils.luv_select(img, channel='l', thresh=(214,255)) # Winner
	# sobel_x = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(24, 255)) # Winner
	# hsv_v = cv_utils.hsv_select(img, channel='v', thresh=(227,255)) # ***Helpful (100,255)
	# hls_s = cv_utils.hls_select(img, channel='s', thresh=(230,255)) # ***Helpful (100,255)
	# binary = np.zeros_like(lab_b)
	# binary[(lab_b == 1) | (luv_l == 1) | (sobel_x == 1) | (hsv_v == 1) | (hls_s == 1)] = 1 # Winner
	# plt.imshow(binary, cmap='gray')
	# plt.show()
	# exit()



	# Binary filters
	lab_b = cv_utils.lab_select(img, channel='b', thresh=(140,255)) # Winner
	luv_l = cv_utils.luv_select(img, channel='l', thresh=(214,255)) # Winner
	sobel_x = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(24, 255)) # Winner
	hsv_v = cv_utils.hsv_select(img, channel='v', thresh=(227,255)) # 
	hls_s = cv_utils.hls_select(img, channel='s', thresh=(230,255)) # ***Helpful (100,255)
	binary = np.zeros_like(lab_b)
	binary[(lab_b == 1) | (luv_l == 1) | (sobel_x == 1) | (hsv_v == 1) | (hls_s == 1)] = 1 # Winner

	if DISPLAY_IMAGE and DISPLAY_PIPELINE:
		plt.imshow(binary, cmap='gray')
		plt.show()

	if SAVE_PIPELINE:
		out_name = 'binary_test4.jpg'
		out_path = os.path.join(OUT_DIR, out_name)
		cv2.imwrite(out_path, binary*255)

	return binary


def undistort_img(img):
	"""
	Function uses the calibration data retrieved in a seperate module and undistorts the camera image
	"""
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)

	if SAVE_PIPELINE:
		out_name = 'undistorted_test4.jpg'
		out_path = os.path.join(OUT_DIR, out_name)
		cv2.imwrite(out_path, cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))

	return undistorted



def get_birds_eye_view(img):
	"""
	Function shifts the perspective of an image into a 'bird's eye view' where it appears as if
	the camera were shooting the road photos from above.
	"""
	width = img.shape[1]
	height = img.shape[0]

	"""
	Define the trapezoid points as percentages just in case we wanted to work
	with different scaled versions of the image
	"""
	topy = 460/720 # 430/720
	boty = (height - 60)/720

	"""
	Define the trapezoid points as percentages just in case we wanted to work
	with different scaled versions of the image
	"""
	toplx = 575/1280 
	toprx = 705/1280 
	botlx = 255/1280
	botrx = 1040/1280 # 1040

	src = np.float32([
		[toplx*width, topy*height], 
		[toprx*width, topy*height],
		[botlx*width, boty*height], 
		[botrx*width, boty*height]])

	# Define some margin on the left and right of the photo for clean visual purposes
	offset = 250 

	dst = np.float32([
		[offset, 0],
		[width - offset, 0],
		[offset, height - 0],
		[width - offset, height - 0]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, (width,height), flags=cv2.INTER_LINEAR)

	if DISPLAY_IMAGE and DISPLAY_PIPELINE:
		plt.imshow(warped, cmap='gray')
		plt.show()

	if SAVE_PIPELINE:
		out_name = 'warped_test4.jpg'
		out_path = os.path.join(OUT_DIR, out_name)
		cv2.imwrite(out_path, warped*255)

	return warped, M, Minv


def find_lanes(img, leftx_base=None, rightx_base=None, rolling_fit=[]):
	"""
	Function does the heavy lifting of identifying the lane pixels and fitting a polynomial
	on that lane
	"""

	"""
	The 'if' statement below is essentially not used at the moment but was put in there
	for later iterations of the project so that we would not have to find an initial base of the lane if
	we had already done so.  Assuming that the next frame would have a similar enough base
	"""
	# Find a starting point for the left and right lanes
	if(leftx_base is None and rightx_base is None):
		histogram = np.sum(img[int(img.shape[0]*2/10):,:], axis=0)

		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((img, img, img))*255
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 150
	# Set minimum number of pixels found to recenter window
	minpix = window_height * margin * 0.25
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	# Mask width
	mask_width = 100

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = img.shape[0] - (window+1)*window_height
	    win_y_high = img.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)

	    # If you found > minpix pixels -> confident about lane so recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Measure curvature
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension (Got metric from lecture)
	xm_per_pix = 3.7/680 # meters per pixel in x dimension (Eyeballed metric from lane spacing in middle of warped image)

	# Fit new polynomials to x,y in world space
	left_fit_real_world = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_real_world = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	y_eval = int(img.shape[0]/2)
	left_curverad = ((1 + (2*left_fit_real_world[0]*y_eval*ym_per_pix + left_fit_real_world[1])**2)**1.5) / np.absolute(2*left_fit_real_world[0])
	right_curverad = ((1 + (2*right_fit_real_world[0]*y_eval*ym_per_pix + right_fit_real_world[1])**2)**1.5) / np.absolute(2*right_fit_real_world[0])
	# Now our radius of curvature is in meters
	print(left_curverad, 'm', right_curverad, 'm')
	# Example values: 632.1 m    626.2 m

	# Visualize
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Find the car's position relative to center
	left_lane_start = left_fitx[img.shape[0]-1]
	right_lane_start = right_fitx[img.shape[0]-1]
	center_lane_start = (left_lane_start + right_lane_start)/2
	car_pos = int(img.shape[1]/2)
	car_offset_pix = (car_pos - center_lane_start)
	car_offset_meters = car_offset_pix*xm_per_pix

	# Create smoother mask
	mask_smooth = np.zeros_like(img).astype(np.uint8)
	color_mask_warp = np.dstack((mask_smooth, mask_smooth, mask_smooth))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_mask_warp, np.int_([pts]), (0,255, 0))


	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# Define indexes for the polynomial fit
	lane_mask = np.zeros(img.shape, dtype=bool)
	for y, left, right in zip(ploty, left_fitx, right_fitx):
		if left >= 0:
			lane_mask[int(y)][int(left)] = True
		if left >= 1:
			lane_mask[int(y)][int(left)-1] = True
		if left >= - 1:
			lane_mask[int(y)][int(left)+1] = True
		if right <= 1280:
			lane_mask[int(y)][int(right)] = True
		if right <= 1281:
			lane_mask[int(y)][int(right)-1] = True
		if right <= 1279:
			lane_mask[int(y)][int(right)+1] = True
	out_img[lane_mask] = [242, 255, 0]

	if SAVE_PIPELINE:
		out_name = 'find_lanes_test4.jpg'
		out_path = os.path.join(OUT_DIR, out_name)
		cv2.imwrite(out_path, out_img)


	if DISPLAY_IMAGE and DISPLAY_PIPELINE:
		plt.imshow(out_img)
		plt.show()
		plt.imshow(color_mask_warp)
		plt.show()	

	return color_mask_warp, left_curverad, right_curverad, car_offset_meters


if __name__ == '__main__':
	main()







