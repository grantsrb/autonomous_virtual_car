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

    ## Note cv2 uses x, y system, not row, col system
    # img_size = canny_edges.shape
    # left_base_col = 20
    # right_base_col = img_size[1]-left_base_col
    # left_upper_col = left_base_col + 130
    # right_upper_col = right_base_col - 130
    # lower_row = img_size[0]
    # upper_row = img_size[0] - 70

    # fill_color = 255
    # fill_img = np.zeros_like(gray_blurred_image)
    # vertices = np.array([[(left_base_col, lower_row), (left_upper_col, upper_row),(right_upper_col, upper_row), (right_base_col, lower_row)]], dtype=np.int32)
    # cv2.fillPoly(fill_img, vertices, fill_color)

    # for i in range(3):
    #     canny_edges[fill_img != 0, i] = 0
    # plt.imshow(canny_edges)
    # plt.show()

    image_copy = cv2.addWeighted(image_copy, 1, canny_edges, 1, 0, image_copy);
    return image_copy;
