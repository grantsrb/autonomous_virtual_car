import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

def show_img(img):
    plt.imshow(img)
    plt.show()

def get_lines(path):
    lines = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

def read_data(lines, root_path, flip=False)
    images = []
    angles = []
    for i,line in enumerate(lines):
        img_path = line[0]
        img_name = img_path.split('/')[-1]
        img_path = root_path + img_name
        img = mpimg.imread(img_path)
        angle = float(line[3])
        images.append(img)
        angles.append(angle)
        if flip:
            images.append(np.fliplr(img)) ## Add image reflected over y-axis
            angles.append(-angle)
    return images, angles


path = './drive_logs/driving_log.csv'
lines = get_lines(path)
images, angles = read_data(lines, './drive_logs/IMG/', flip=True)

path = './drive_logs/driving_log1.csv'
lines = lines + get_lines(path)
images1, angles1 = read_data(lines, './drive_logs/IMG1/', flip=True)

path = './drive_logs/driving_log2.csv'
lines = lines + get_lines(path)
images2, angles2 = read_data(lines, './drive_logs/IMG2/', flip=True)

images = images + images2 + images3
del images2
del images3
angles = angles + angles2 + angles3
del angles2
del angles3


X_train, y_train = np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)
print(X_train.shape)
print(y_train.shape)

X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, concatenate, \
        Flatten, Dropout, Lambda

towers = []
conv_shapes = [1,3,5]
depths = [6,8,8,6]

input_shape = tuple([X_train.shape[x] for x in range(1,len(X_train.shape))])
print(input_shape)
inputs = Input(shape=input_shape)
zen_layer = Lambda(lambda x: x/255.0 - 0.5)(inputs) # Normalize and center images

for shp in conv_shapes:
    towers.append(Conv2D(depths[0], (shp, shp), padding='same', activation='elu')(zen_layer))

layer = concatenate(towers, axis=3)
maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)

for i in range(1,len(depths)):
    towers = []
    for shp in conv_shapes:
        towers.append(Conv2D(depths[i], (shp, shp), padding='same', activation='elu')(maxPool))

    layer = concatenate(towers, axis=3)
    maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)

flat_layer = Flatten()(maxPool)
flat_layer = Dropout(.5)(flat_layer)

fc_layer = Dense(100, activation='elu')(flat_layer)
fc_layer = Dense(25, activation='elu')(fc_layer)
output = Dense(1)(fc_layer)

model = Model(inputs=inputs, outputs=output)
model.load_weights('./model.h5')
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=2, batch_size=128, validation_split=0.3, shuffle=True)

model.save('model.h5')





##
