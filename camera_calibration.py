import numpy as np
import cv2
import glob

# Save directory for camera data
savedir="camera_data/"

# using a 9 x 6 chessboard for calibration
pattern_rows = 9
pattern_columns = 6 

# each square is 23.81 x 23.81 
square_size = 23.8125 #mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# z is 0 since points are obtained from a flat surface 

objp = np.zeros((pattern_columns*pattern_rows,3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_rows,0:pattern_columns].T.reshape(-1,2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Obtain all sample images
images = glob.glob('calib_images/*.jpg')

# Iterate from each image
for image in images:
    img = cv2.imread(image)
    
    # Converts image to gray scale for corner detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (pattern_rows,pattern_columns),None)

    # If corners were found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Optimizes corner detection
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (pattern_rows,pattern_columns), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        
    img1 = img
        
cv2.destroyAllWindows()

print("----------------------")
print("Calibrating...")
ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix")
print(cam_mtx)
np.save(savedir+'cam_mtx.npy', cam_mtx)

print("Distortion Coeff")
print(dist)
np.save(savedir+'dist.npy', dist)

print("r vecs")
print(rvecs[2])

print("t Vecs")
print(tvecs[2])

print("Calibration ended")
print("----------------------")

h, w = img1.shape[:2]
print("Image Width, Height")
print(w, h)

newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w,h), 1, (w,h))

print("Region of Interest")
print(roi)
np.save(savedir+'roi.npy', roi)

print("New Camera Matrix")
print(newcam_mtx)
np.save(savedir+'newcam_mtx.npy', newcam_mtx)

# undistort
undst = cv2.undistort(img1, cam_mtx, dist, None, newcam_mtx)

cv2.imshow('img1', img1)
cv2.waitKey(5000)      
cv2.destroyAllWindows()
cv2.imshow('img1', undst)
cv2.waitKey(5000)      
cv2.destroyAllWindows()

