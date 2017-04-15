import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

def show_img(img):
    plt.imshow(img)
    plt.show()

CROP_SIZE = 50

############ Data Section

def get_lines(path):
    lines = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

def read_data(lines, images, angles, root_path, add_flip=False, add_half_flip=False, only_flip=False):
    for i,line in enumerate(lines):
        img_path = line[0]
        img_name = img_path.split('/')[-1]
        img_path = root_path + img_name
        img = mpimg.imread(img_path)
        img = img[CROP_SIZE:] # Crops out upper parts of img
        angle = float(line[3])
        if not only_flip:
            images.append(img)
            angles.append(angle)
        if (add_half_flip and i % 2 == 0) or add_flip or only_flip:
            images.append(np.fliplr(img)) ## Add image reflected over y-axis
            angles.append(-angle)
    return images, angles

images = []
angles = []

path = './improved_data/centered_log.csv'
lines = get_lines(path)
print("Begin reading images set 1")
images, angles = read_data(lines, images, angles, './improved_data/centered/', add_flip=True)

path = './improved_data/corrections_log.csv'
lines = get_lines(path)
print("Begin reading images set 2")
images, angles = read_data(lines, images, angles, './improved_data/corrections/')

path = './improved_data/curves_log.csv'
lines = get_lines(path)
print("Begin reading images set 3")
images, angles = read_data(lines, images, angles, './improved_data/curves/', add_flip=True)

print("Begin conversion to numpy")
X_train = np.array(images, dtype=np.float32)
del images
y_train = np.array(angles, dtype=np.float32)
del angles


print("Features Shape: " + str(X_train.shape))
print("Labels Shape: " + str(y_train.shape))

print("Begin shuffle")
X_train, y_train = shuffle(X_train, y_train)


################ Keras Section
print("Begin Keras Imports")
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, concatenate, \
        Flatten, Dropout, Lambda

print("Begin model construction")
towers = []
conv_shapes = [1,3,5]
depths = [6,8,8,8,6]

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
    if i % 2 == 0:
        layer = Dropout(.1)(layer)
    maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)

flat_layer = Flatten()(maxPool)
flat_layer = Dropout(.4)(flat_layer)

fc_layer = Dense(100, activation='elu')(flat_layer)
fc_layer = Dense(25, activation='elu')(fc_layer)
output = Dense(1)(fc_layer)

model = Model(inputs=inputs, outputs=output)
print("Begin load weights")
model.load_weights('./cropped_5convs.h5')
print("Begin model compile")
model.compile(loss='mse', optimizer='adam')
print("Begin model fit")
model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.25, shuffle=True)

model.save('cropped_5convs.h5')





##
