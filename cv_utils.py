import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import pickle





def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
    	return None
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return binary


def mag_sobel_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Function that applies Sobel x and y, 
	then computes the direction of the gradient
	and applies a threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(grad_dir)
    binary[(grad_dir > thresh[0]) & (grad_dir<thresh[1])] = 1
    
    return binary


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, channel='h', thresh=(0, 255)):
    
    channels = {'h':0, 'l':1, 's':2}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,channels[channel]]
    binary = np.zeros_like(channel)
    binary[(channel>thresh[0]) & (channel<thresh[1])]=1
    
    return binary


# Define a function that thresholds the S-channel of HSV
# Use exclusive lower bound (>) and inclusive upper (<=)
def hsv_select(img, channel='h', thresh=(0, 255)):
    
    channels = {'h':0, 's':1, 'v':2}
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channel = hsv[:,:,channels[channel]]
    binary = np.zeros_like(channel)
    binary[(channel>thresh[0]) & (channel<thresh[1])]=1
    
    return binary


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def rgb_select(img, channel='r', thresh=(0, 255)):
    
    channels = {'r':0, 'g':1, 'b':2}
    channel = img[:,:,channels[channel]]
    binary = np.zeros_like(channel)
    binary[(channel>thresh[0]) & (channel<thresh[1])]=1
    
    return binary


def rgb_white_thresh(img, thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray>thresh[0]) & (gray<thresh[1])]=1
    
    return binary


def rgb_yellow_thresh(img, thresh=(0, 255)):
    
    red = img[:,:,0]
    green = img[:,:,1]
    binary = np.zeros_like(red)
    binary[(red>thresh[0]) & (red<thresh[1]) & (green>thresh[0]) & (green<thresh[1])]=1
    
    return binary