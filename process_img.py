import numpy as np
import glob
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv_utils


TEST_DIR = 'test_images'
GLOB_EXT = 'test*.jpg'
TEST_NAME = 'test1.jpg'
STRAIGHT = 'straight_lines1.jpg'
CURVED = 'test2.jpg'

with open('calibration.p', 'rb') as pickle_in:
	dist_pickle = pickle.load(pickle_in)

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

def main():

	# img = mpimg.imread(os.path.join(TEST_DIR, TEST_NAME))
	# undistorted = undistort_img(img)
	# binary_img = get_binary_img(undistorted)

	# # Visualize binary filters
	# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
	# ax1.set_title('1')
	# # ax1.imshow(comb)
	# ax1.imshow(comb, cmap='gray')
	# ax2.set_title('2')
	# ax2.imshow(img, cmap='gray')
	# ax3.set_title('3')
	# ax3.imshow(comb1, cmap='gray')
	# ax4.set_title('4')
	# ax4.imshow(comb2, cmap='gray')
	# plt.show()

	img = mpimg.imread(os.path.join(TEST_DIR, CURVED))
	undistorted = undistort_img(img)
	binary = get_binary_img(undistorted)
	birds_eye, M, Minv = get_birds_eye_view(binary)
	find_lines(birds_eye)



def get_binary_img(img):

	# # Filtering for Colors white and yellow
	# thresh_w = cv_utils.rgb_white_thresh(undistorted, thresh=(200,255)) # ***Helpful (200, 255)
	# thresh_w2 = cv_utils.rgb_white_thresh(undistorted, thresh=(195,255)) # Good (200, 255)

	# # Filtering for S channel then taking sobel of that image to clean it up
	hls_s = cv_utils.hls_select(img, channel='s', thresh=(100,255)) # ***Helpful (100,255)
	hsv_s = cv_utils.hsv_select(img, channel='s', thresh=(125,255)) # ***Helpful (125,255)
	# channel_comb = np.zeros_like(hls_s) 
	# channel_comb[(hls_s == 1) | (hsv_s == 1)] = 1 # Winner


	# kernal test
	sobelx = cv_utils.abs_sobel_thresh(img, orient='x', sobel_kernel=11, thresh=(10, 255))
	sobely = cv_utils.abs_sobel_thresh(img, orient='y', sobel_kernel=11, thresh=(10, 255)) # ***Good
	# sobel_comb = np.zeros_like(sobelx)
	# sobel_comb[(sobelx==1) & (sobely==1)] = 1

	binary = np.zeros_like(hls_s)
	binary[(hls_s == 1) | (hsv_s == 1) | ((sobelx==1) & (sobely==1))] = 1 # Winner

	# # Visualize
	# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
	# ax1.set_title('reg')
	# ax1.imshow(img, cmap='gray')
	# ax2.set_title('warp')
	# ax2.imshow(channel_comb, cmap='gray')
	# ax3.set_title('warp')
	# ax3.imshow(comb, cmap='gray')
	# plt.show()
	# exit()

	return binary


def undistort_img(img):
	return cv2.undistort(img, mtx, dist, None, mtx)


def get_birds_eye_view(img):
	width = img.shape[1]
	height = img.shape[0]

	# print("Dims:", width, height)

	topy = 460/720 # 430/720
	boty = (height - 60)/720

	toplx = 570/1280 # 580/1280 # 623/1280
	toprx = 705/1280 # 705/1280 # 650/1280
	botlx = 255/1280
	botrx = 1040/1280

	src = np.float32([
		[toplx*width, topy*height], 
		[toprx*width, topy*height],
		[botlx*width, boty*height], 
		[botrx*width, boty*height]])

	offset = 250 # 0.15 * height

	dst = np.float32([
		[offset, 0],
		[width - offset, 0],
		[offset, height - 0],
		[width - offset, height - 0]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, (width,height), flags=cv2.INTER_LINEAR)

	# # Visualize
	# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	# ax1.set_title('reg')
	# ax1.imshow(img, cmap='gray')
	# ax2.set_title('warp')
	# ax2.imshow(warped, cmap='gray')
	# plt.show()

	return warped, M, Minv


def find_lines(img):
	histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
	# plt.plot(histogram)
	# plt.show()

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
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

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
	    # If you found > minpix pixels, recenter next window on their mean position
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


	# Visualize
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()


if __name__ == '__main__':
	main()







