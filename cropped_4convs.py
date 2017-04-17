import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import canny_augment as canny
import time

def show_img(img):
    plt.imshow(img)
    plt.show()

CROP_SIZE = 50

############ Data Reading Functions

def get_lines(path):
    lines = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

def read_data(lines, images, angles, root_path, add_edges=True, add_flip=False, add_half_flip=False, only_flip=False):
    for i,line in enumerate(lines):
        img_path = line[0]
        img_name = img_path.split('/')[-1]
        img_path = root_path + img_name
        img = mpimg.imread(img_path)
        if add_edges:
            img = canny.add_edges(img)
        img = img[CROP_SIZE:] # Crops out upper parts of img
        angle = float(line[3])
        if not only_flip:
            images.append(img)
            angles.append(angle)
        if (add_half_flip and i % 2 == 0) or add_flip or only_flip:
            images.append(np.fliplr(img)) ## Add image reflected over y-axis
            angles.append(-angle)
    return images, angles


############ Data Reading

logs_and_imgs_folder = 'improved_data'
log_names = ['t2_centered_log','t2_corrections_log','t2_curves_log','t2_reinforcement_log']
IMG_folder_names = ['t2_centered','t2_corrections','t2_curves','t2_reinforcement']
add_flips = [True, False, True, False]
add_half_flips = [False, True, False, True]

images = []
angles = []

base_time = time.time()

# Reads in images, crops images, and applies canny edge detection to images
for log,img_folder,full,half in zip(log_names, IMG_folder_names, add_flips, add_half_flips):
    path = './'+logs_and_imgs_folder+'/' + log + '.csv'
    lines = get_lines(path)
    print("Begin reading " + img_folder)
    images, angles = read_data(lines, images, angles, './'+logs_and_imgs_folder+'/'+ img_folder+'/', add_flip=full, add_half_flip=half)
    print("Total images: " + str(len(images)))
    print("Running Time: " + str(int(time.time() - base_time))+"s")


print("Begin conversion to numpy")
X_train = np.array(images, dtype=np.float32)
del images
y_train = np.array(angles, dtype=np.float32)
del angles
print("Running Time: " + str(int(time.time() - base_time))+"s")


print("Features Shape: " + str(X_train.shape))
print("Labels Shape: " + str(y_train.shape))

print("Begin shuffle")
X_train, y_train = shuffle(X_train, y_train)
print("Running Time: " + str(int(time.time() - base_time))+"s")


################ Keras Section

print("Begin Keras Imports")
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, concatenate, \
        Flatten, Dropout, Lambda

print("Begin model construction")
stacks = []
conv_shapes = [1,3,5] # parallel filter sizes
depths = [6,8,8,6] # output depth at each convolutional layer

input_shape = tuple([X_train.shape[x] for x in range(1,len(X_train.shape))])
print(input_shape)
inputs = Input(shape=input_shape)
zen_layer = Lambda(lambda x: x/255.0 - 0.5)(inputs) # Normalizes and centers images

# First convolutional layer runs filters of conv_shapes sizes on the preprocessed inputs
# The outputs are then concatenated into a single layer
for shp in conv_shapes:
    stacks.append(Conv2D(depths[0], (shp, shp), padding='same', activation='elu')(zen_layer))

layer = concatenate(stacks, axis=3)
maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)

# The following convolutional layers each share the stacking-parallel shape
for i in range(1,len(depths)):
    stacks = []
    for shp in conv_shapes:
        stacks.append(Conv2D(depths[i], (shp, shp), padding='same', activation='elu')(maxPool))

    layer = concatenate(stacks, axis=3)
    maxPool = MaxPooling2D((2,2), strides=(2,2),padding='valid')(layer)

flat_layer = Flatten()(maxPool)
flat_layer = Dropout(.5)(flat_layer) # Dropout reduces overfitting

# Fully connected layers
fc_layer = Dense(100, activation='elu')(flat_layer)
fc_layer = Dense(25, activation='elu')(fc_layer)
output = Dense(1)(fc_layer)

print("Running Time: " + str(int(time.time() - base_time))+"s")
model = Model(inputs=inputs, outputs=output)
print("Begin load weights")
model.load_weights('./cropped_4convs_t2.h5')
print("Begin model compile")
model.compile(loss='mse', optimizer='adam')
print("Begin model fit")
model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.25, shuffle=True)

model.save('cropped_4convs_t2.h5')
print("Total Running Time: " + str((time.time() - base_time)//60) + "mins")





##
