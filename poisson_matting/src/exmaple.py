import numpy as np
import cv2 as cv

# Load a color image in grayscale
img = cv.imread('pic.jpg', 1)   # 1, 0, -1 for color, grey scale and uncolored?

# display an image
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
