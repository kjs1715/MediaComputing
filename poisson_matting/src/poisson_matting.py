# # coding=utf-8

import sys
import os
import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# global

fileNum = 1

# threshold for gauss-seidal model
th = 0.1

def getFlag(trimap, gray, height, width):
    # sign for seperate foreground and background
    fGray = np.zeros((height, width), dtype=np.uint8)
    bGray = np.zeros((height, width), dtype=np.uint8)
    unknown = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if trimap[i,j] == 0:
                bGray[i,j] = 1
            elif trimap[i,j] == 255:
                fGray[i,j] = 1
            else:
                unknown[i,j] = 1
    fImg = fGray * gray
    bImg = bGray * gray
    fInpaint = cv2.inpaint(fImg, (unknown + bGray), 3, cv2.INPAINT_TELEA)
    bInpaint = cv2.inpaint(bImg, (unknown + fGray), 3, cv2.INPAINT_TELEA)

    cv2.imshow('fInpaint', fInpaint)
    cv2.imshow('bInpaint', bInpaint)
    cv2.waitKey(0)

    fInpaint = fInpaint * np.logical_not(bGray)
    bInpaint = bInpaint * np.logical_not(fGray)
    # print(bInpaint)
    # print(np.logical_not(fGray))
    # print(np.logical_not(fGray).astype(np.float32))
    # print(bInpaint*np.logical_not(fGray))
    # print(bInpaint*np.logical_not(fGray).astype(np.float32))
    # print(bInpaint*np.logical_not(fGray).astype(np.uint8))

    cv2.imshow('fInpaint', fInpaint)
    cv2.imshow('bInpaint', bInpaint)
    cv2.waitKey(0)

    diff = scipy.ndimage.filters.gaussian_filter(fInpaint - bInpaint, 0.2)

    # diff = cv.GaussianBlur(fInpaint - bInpaint, (3,3), 0)
    plt.imshow(diff)
    cv2.waitKey(0)

    return fGray, bGray, diff, unknown

def x_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def y_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def getLaplace(gray, diff):
    # xGrad = x_gradient(gray)
    # yGrad = y_gradient(gray)
    # yyGrad, none = np.gradient(yGrad / diff)
    # none, xxGrad = np.gradient(xGrad / diff)
    yGrad, xGrad = np.gradient(gray)
    yyGrad, none = np.gradient(yGrad / diff) 
    none, xxGrad = np.gradient(xGrad / diff)
    laplace = xxGrad + yyGrad
    # print(laplace)
    return laplace

def getAlpha(laplace, alpha, unknown):
    height = unknown.shape[0]
    width = unknown.shape[1]
    alphaNew = alpha.copy()
    alphaOld = np.zeros((height, width), dtype=np.uint8)
    n = 1
    while (n < 100 and np.sum(np.abs(alphaNew - alphaOld)) > th):
        alphaOld = alphaNew.copy()
        # print(alphaOld)
        for i in range(1, height-1):
            for j in range(1, width-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4 * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - laplace[i,j])
        n += 1
    alpha = alphaNew
    return alphaNew

def run(fileName):
    global fileNum
    srcFile = fileName[0]
    trimapFile = fileName[1]

    raw = scipy.misc.imread(srcFile)
    gray = scipy.misc.imread(srcFile, flatten='True')
    trimap = scipy.misc.imread(trimapFile, flatten='True')

    # print(raw)
    height = raw.shape[0]
    width = raw.shape[1]

    fGray, bGray, diff, unknown = getFlag(trimap, gray, height, width)
    laplace = getLaplace(gray, diff)
    # initialize alpha = 1, beta = 0, unknown = 0.5
    estimate = unknown * 0.5 + fGray
    alpha = getAlpha(laplace, estimate, unknown)
    alpha = np.minimum(np.maximum(alpha,0),1).reshape(height, width)
    # print(alpha)
    plt.imshow(alpha)
    scipy.misc.imsave(str(fileNum) + '1.png', alpha)
    fileNum += 1
    print('ddd')
    # plt.show()
    
if __name__ == '__main__':
    for fileName in [('troll.png', 'trollTrimap.bmp'), ('dog.png','dog_trimap.bmp'), ('can.png', 'can_trimap.bmp')]:
        run(fileName)