import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

def show_img(img):
    plt.imshow(img)
    plt.show()

def inversion_transform(img_shape):
    m_shape = (img_shape[1],img_shape[1])
    m = np.zeros(m_shape, dtype=np.float32)
    for i,row in enumerate(m):
        m[i][-i] = 1
    return m

def reflect_img(img, m):
    flipped_img = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[-1]):
        flipped_img[:,:,i] = np.dot(img[:,:,i], m)
    return flipped_img

def get_lines(path):
    lines = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

path = './drive_logs/driving_log1.csv'
lines = get_lines(path)

img = mpimg.imread('./drive_logs/IMG1/' + lines[0][0].split('/')[-1])
show_img(img)
show_img(np.fliplr(img))
# flip_matrix = inversion_transform(img.shape)
#
# def read_data(lines, root_path)
#     images = []
#     angles = []
#     for i,line in enumerate(lines):
#         img_path = line[0]
#         img_name = img_path.split('/')[-1]
#         img_path = root_path + img_name
#         img = mpimg.imread(img_path)
#         angle = float(line[3])
#         images.append(img)
#         angles.append(angle)
#         images.append(reflect_img(img, flip_matrix)) ## Add image reflected over y-axis
#         angles.append(-angle)
#     return np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)
#
# X_train, y_train = read_data(lines, './drive_logs/IMG1/')
#
# print(X_train.shape)
# print(y_train.shape)
#
# X_train, y_train = shuffle(X_train, y_train)
#
# from keras.models import Sequential, Model
# from keras.layers import Conv2D, MaxPooling2D, Dense, Input, concatenate, \
#         Flatten, Dropout, Lambda
#
# epochs = 10
# towers = []
# conv_shapes = [1,3,5]
# depths = [6,8,8,6]
#
# input_shape = tuple([X_train.shape[x] for x in range(1,len(X_train.shape))])
# print(input_shape)
# inputs = Input(shape=input_shape)
# zen_layer = Lambda(lambda x: x/255.0 - 0.5)(inputs) # Normalize and center images
#
# for shp in conv_shapes:
#     towers.append(Conv2D(depths[0], (shp, shp), padding='same', activation='elu')(zen_layer))
#
# layer = concatenate(towers, axis=3)
# maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)
#
# for i in range(1,len(depths)):
#     towers = []
#     for shp in conv_shapes:
#         towers.append(Conv2D(depths[i], (shp, shp), padding='same', activation='elu')(maxPool))
#
#     layer = concatenate(towers, axis=3)
#     maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)
#
# flat_layer = Flatten()(maxPool)
# flat_layer = Dropout(.5)(flat_layer)
#
# fc_layer = Dense(100, activation='elu')(flat_layer)
# fc_layer = Dense(25, activation='elu')(fc_layer)
# output = Dense(1)(fc_layer)
#
# model = Model(inputs=inputs, outputs=output)
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, nb_epoch=epochs, batch_size=128, validation_split=0.3, shuffle=True)
#
# model.save('model.h5')
#
#
#
# ##
