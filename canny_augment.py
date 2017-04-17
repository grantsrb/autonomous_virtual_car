import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def process_with_path(path):
    return add_edges(mpimg.imread(path));

def add_edges(image):
    image_copy = np.copy(image);
    gray_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY);

    kernel_size = 3;
    gray_blurred_image = cv2.GaussianBlur(gray_copy, (kernel_size, kernel_size), 0);

    low_thresh = 30;
    hi_thresh = 160;
    canny_edges = cv2.Canny(gray_blurred_image, low_thresh, hi_thresh)
    canny_edges = np.array([canny_edges for i in range(3)])
    canny_edges = np.transpose(canny_edges, [1,2,0])
    canny_edges[canny_edges[:,:,1] != 0, 1] = 0

    image_copy = cv2.addWeighted(image_copy, 1, canny_edges, 1, 0, image_copy);
    return image_copy;
