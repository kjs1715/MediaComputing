import os
import sys
import numpy as np
import cv2
import time
import tqdm 
import imghdr

FILE_PATH = '/Users/kim/Desktop/Git/ImageMatting/seam_carving/'
OUTPUT_PATH = '/Users/kim/Desktop/Git/ImageMatting/seam_carving/'

def x_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def y_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def get_energy(way, img):
    # print(img)
    # print("img", img.shape)
    height, width, _ = img.shape
    # gauss_blur = cv2.GaussianBlur(img, (3, 3), 0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = x_gradient(gray)
    dy = y_gradient(gray)
    energy = cv2.add(np.absolute(dx), np.absolute(dy))
    # print(energy.shape)
    M = np.zeros((height, width))

    if way == 0:
        # vertical
        for i in range(1, height):
            for j in range(width):
                if j == 0:
                    M[i, j] = energy[i, j] + min(M[i-1, j], M[i-1, j+1], M[i-1, j+2])
                elif j == width-1:
                    M[i, j] = energy[i, j] + min(M[i-1, j-2], M[i-1, j-1], M[i-1, j])
                else:
                    M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
    else:
        # horizontal
        for j in range(1, width):
            for i in range(0, height):
                if i == 0:
                    M[i, j] = energy[i, j] + min(M[i, j-1], M[i+1, j-1], M[i+2, j-1])
                elif i == height-1:
                    M[i, j] = energy[i, j] + min(M[i-2, j-1], M[i-1, j-1], M[i, j-1])
                else:
                    M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i, j-1], M[i+1, j-1])

    return M

'''
    :para
        seam_mask -> (height, width), dtype=np.bool
        seam -> [(last_min, j)] or [(i, last_min)]
        seam_count -> count for seam, if not 0, use counter for positioning other pixels
'''
def save_seam(way, seam_mask, seam, seam_count):
    height, width = seam_mask.shape
    # print("seam_count:", seam_count)
    if way == 0:
        if seam_count == 0:
            for i, j in seam:
                seam_mask[i, j] = True
        else:
            for i, j in seam:
                # print("i and last_min:", i, j)
                k = 0
                counter = -1
                for k in range(width):
                    # print("iteration :" , k, seam_mask[i, k])
                    if seam_mask[i, k] == False:
                        counter += 1
                    if counter == j:
                        seam_mask[i, k] = True
                        break    

                # print("j : ", j)
                # print("k : ", k)
                # print("counter:", counter)
                # time.sleep(1)
    else:
        if seam_count == 0:
            for i, j in seam:
                seam_mask[i, j] = True
        else:
            for i, j in seam:
                k = 0
                counter = -1
                for k in range(height):
                    if seam_mask[k, j] == False:
                        counter += 1
                    if counter == i:
                        seam_mask[k, j] = True
                        break

    #     else:
    
    return seam_mask

def get_seam(way, M, img):
    seam = []
    height, width, _ = img.shape
    if way == 0:
        last_min = np.argmin(M[height-1, :])
        seam.append([height-1, last_min])
        for i in reversed(range(height-1)):
            if last_min == 0:
                last_min = last_min + np.argmin([M[i, last_min], M[i, last_min+1]])
            elif last_min == width-1:
                last_min = last_min + np.argmin([M[i, last_min-1], M[i, last_min]]) - 1
            else:
                # print(M[i, last_min-1],  M[i, last_min], M[i, last_min+1])
                last_min = last_min + np.argmin([M[i, last_min-1], M[i, last_min], M[i, last_min+1]]) - 1
            # print(last_min)
            seam.append([i, last_min])

    else:
        # horizontal
        last_min = np.argmin(M[:, width-1])
        seam.append((last_min, width-1))
        for j in reversed(range(width-1)):
            if last_min == 0:
                last_min = last_min + np.argmin([M[last_min, j], M[last_min+1, j]])
            elif last_min == height-1:
                last_min = last_min + np.argmin([M[last_min-1, j], M[last_min, j]]) - 1
            else:
                last_min = last_min + np.argmin([M[last_min-1, j], M[last_min, j], M[last_min+1, j]]) - 1
            seam.append((last_min, j))
    return seam

def get_seams(way, img, scale):
    carve_img = img
    height, width, channel = img.shape
    seam_mask = np.zeros((height, width), dtype=np.bool)

    if way == 0:
        # vertical
        seam_count = int(width * scale)
        for i in tqdm.trange(seam_count):
            M = get_energy(way, carve_img)
            seam = get_seam(way, M, carve_img)
            seam_mask = save_seam(way, seam_mask, seam, i)
            carve_img = carve(way, M, carve_img, seam)

        # enlarge

    else:
        seam_count = int(height * scale)
        for i in tqdm.trange(seam_count):
            M = get_energy(way, carve_img)
            seam = get_seam(way, M, carve_img)
            seam_mask = save_seam(way, seam_mask, seam, i)
            carve_img = carve(way, M, carve_img, seam)


    return seam_mask, seam_count
'''
    find minimum seam to carve
    :para
        way = 0 : vertical
        way = 1 : horizontal
'''
def carve(way, M, img, seam):
    # print("carve_img:", img.shape)
    img_height, img_width, img_channel = img.shape
    if way == 0:
        # vertical
        # need to initialize first value at the bottom

        # print(last_min)

        # remove seam that we dont need (vertical)
        new_img = np.zeros((img_height, img_width-1, img_channel), np.uint8)
        for i, j in seam:
            for k in range(j):
                new_img[i, k] = img[i, k]
            for k in range(j, img_width-1):
                new_img[i, k] = img[i, k+1]                
            # new_img[y, 0:x] = img[y, 0:x]
            # new_img[y, x:img_width - 1] = img[y, x + 1:img_width]

    else: 

        new_img = np.zeros((img_height-1, img_width, img_channel), np.uint8)
        for i, j in seam:
            for k in range(i):
                new_img[k, j] = img[k, j]
            for k in range(i, img_height-1):
                new_img[k, j] = img[k+1, j]
            # new_img[0:y, x] = img[0:y, x]
            # new_img[y:img_height - 1, x] = img[y + 1:img_height, x]
    # print('carved:', new_img.shape)
    # cv2.imshow(new_img)

    return new_img
    
def carve_img(new_image, new_width, new_height, width, height):
    for i in tqdm.trange(width - new_width):
        M = get_energy(0, new_image)
        seam = get_seam(0,  M, new_image)
        new_image = carve(0, M, new_image, seam)
    for i in tqdm.trange(height - new_height):
        M = get_energy(1, new_image)
        seam = get_seam(1, M, new_image)
        new_image = carve(1, M, new_image, seam)

    return new_image

def change_to_int(seam):

    r, c = seam.shape
    s = np.zeros((r, c), np.uint8)
    for i in range(r):
        for j in range(c):
            if seam[i, j] == True:
                s[i, j] = 255

    cv2.imshow('seam', s)
    cv2.waitKey(0)

def enlarge_img(way, img, width, height, scale):
    # percentage for seam num
    seams_percentage = scale
    img_height, img_width, img_channel = img.shape

    if way == 0:
    # change_to_int(seam_mask)'
        seam_mask, seam_count = get_seams(0, img, seams_percentage)
        new_image = np.zeros((img_height, img_width + seam_count , img_channel), np.uint8)
        for i in tqdm.trange(img_height):
            p = 0

            for j in range(img_width):
                if seam_mask[i, j] == True:
                    new_image[i, p] = img[i, j]
                    p += 1
                    new_image[i, p] = img[i, j]
                else:
                    new_image[i, p] = img[i, j]
                p += 1
    else:
        seam_mask, seam_count = get_seams(1, img, seams_percentage)
        new_image = np.zeros((img_height + seam_count, img_width , img_channel), np.uint8)
        for j in tqdm.trange(img_width):
            p = 0
            for i in range(img_height):
                if seam_mask[i, j] == True:
                    new_image[p, j] = img[i, j]
                    p += 1
                    new_image[p, j] = img[i, j]
                else:
                    new_image[p, j] = img[i, j]
                p += 1

    # cv2.imshow('new_image', new_image)
    # cv2.waitKey(0)
    return new_image

def check_img(new_img):
    r, c, _ = new_img.shape
    for i in range(r):
        for j in range(c):
            if new_img[i, j].all() == 0:
                print(i, j)

def run(oper, img, width, height, file):    
    new_image = img
    if oper == 0:
        print("Input new size...width: {0}, height: {1}".format(width, height))
        new_width = int(input("Width:"))
        new_height = int(input("Height:"))
        new_image = carve_img(new_image, new_width, new_height, width, height)
    else:
        print("width: {0}, height: {1}".format(width, height))
        way = int(input("Choose direction: 1. Vertical  2. Horizontal"))
        scale = float(input("Input scale(0.0~1.0):"))
        new_image = enlarge_img(way-1, new_image, width, height, scale)
    # check_img(new_image)
    cv2.imwrite(OUTPUT_PATH+file.split('.')[0]+'_resized.jpg', new_image)
    # cv2.imshow('new_image', new_image)
    cv2.waitKey(0)
     
if __name__ == '__main__':
    for file in ['beach.jpg', 'waterfall.jpg', 'castle.jpg']:
        img = cv2.imread(FILE_PATH+file)
        # print(img.shape)
        height = img.shape[0]
        width = img.shape[1]
        # print(width, height)
        print("File name:", file)
        print("Choose : 1. Carve  2. Enlarge")
        oper = int(input("Num:"))
        
        # print(new_image.shape)
        # if new_width > width or new_height > height:
        #     print("Pls input correct new size!...")
        #     sys.exit(0)
        # if oper == 0
        run(oper-1, img, width, height, file)
    


    