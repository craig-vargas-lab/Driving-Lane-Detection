import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import glob
import os


CAL_DIR = 'camera_cal'
CAL_FILE = 'camera_cal/calibration1.jpg'
GLOB_EXT = 'calibration*.jpg'
OUT_DIR = 'output_images'
OUT_PREFIX = 'findCorners'
OUT_SUFFIX = '.jpg'

SAVE_DRAW_CORNERS = False

UNDISTORT = True
UNDISTORT_FILE = 'camera_cal/calibration1.jpg'


image_paths = glob.glob(os.path.join(CAL_DIR, GLOB_EXT))

# img = mpimg.imread(CAL_FILE)
# plt.imshow(img)
# plt.show()

objpoints = []
imgpoints =[]

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

num_corners = (9, 6)
success_count = 0

for idx, path in enumerate(image_paths):
	print('Currently working on:', path)
	img = mpimg.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

	if ret == True:
		success_count += 1
		imgpoints.append(corners)
		objpoints.append(objp)

		if SAVE_DRAW_CORNERS:
			img = cv2.drawChessboardCorners(img, num_corners, corners, ret)
			out_name = OUT_PREFIX + str(idx + 1) + OUT_SUFFIX
			out_path = os.path.join(OUT_DIR, out_name)
			cv2.imwrite(out_path, img)
			# plt.imshow(img)
			# plt.show()

print()
print("Success count: ", success_count)


img = cv2.imread(CAL_FILE)
formatted_shape = (img.shape[1], img.shape[0])


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, formatted_shape, None, None)

if UNDISTORT:
	img = mpimg.imread(UNDISTORT_FILE)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	out_name = 'undistorted_calibration1.jpg'
	out_path = os.path.join(OUT_DIR, out_name)
	cv2.imwrite(out_path, undist)
	plt.imshow(undist)
	plt.show()
	exit()


dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
with open ('calibration.p', 'wb') as pickle_out:
	pickle.dump(dist_pickle, pickle_out)

