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

def read_data_line(line, folder, flip):
    img_path = line[0]
    img_name = img_path.split('/')[-1]
    root_path = './'+data_folder+'/'+ folder+'/'
    img_path = root_path + img_name
    img = mpimg.imread(img_path)
    img = img[CROP_SIZE:] # Crops out upper parts of img
    img = canny.add_edges(img)
    angle = float(line[3])
    if flip:
        img = np.fliplr(img) ## Add image reflected over y-axis
        angle = -angle
    return img, angle

def get_generator_lines(data_folder, log_names, IMG_folders):
    lines = []
    folders = []
    flips = []
    for log,img_folder in zip(log_names, IMG_folders):
        path = './'+data_folder+'/' + log + '.csv'
        new_lines = get_lines(path)
        lines = lines + new_lines + new_lines
        for flip in [True,False]:
            for i in range(len(new_lines)):
                folders.append(img_folder)
                flips.append(flip)
    return lines,folders,flips

# Generator for reading and cropping images, and applying canny edge detection to images
def data_generator(data_folder, log_names, IMG_folders, batch_size=128):
    lines,folders,flips = get_generator_lines(data_folder,log_names,IMG_folders)
    while True:
        lines,folders,flips = shuffle(lines,folders,flips)
        for batch in range(0, len(lines), batch_size):
            images = []
            angles = []
            for line,folder,flip in zip(lines[batch:batch+batch_size], folders[batch:batch+batch_size], flips[batch:batch+batch_size]):
                image, angle = read_data_line(line, folder, flip)
                images.append(image)
                angles.append(angle)
            images, angles = np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)
            images, angles = shuffle(images, angles)
            yield (images, angles)

def epoch_steps(data_folder,log_names, batch_size=128):
    total = 0
    for log in log_names:
        path = './'+data_folder+'/' + log + '.csv'
        lines = get_lines(path)
        total += len(lines)
    if (2*total) % batch_size == 0: return (2*total)//batch_size
    return (2*total)//batch_size+1

############ Data Reading

base_time = time.time()

data_folder = 'improved_data'
log_names = ['t2_centered_log','t2_corrections_log','t2_curves_log','t2_reinforcement_log', 'centered_log']
IMG_folders = ['t2_centered','t2_corrections','t2_curves','t2_reinforcement', 'centered']

n_steps = epoch_steps(data_folder, log_names)

train_generator = data_generator(data_folder, log_names, IMG_folders)


################ Keras Section

print("Begin Keras Imports")
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, concatenate, \
        Flatten, Dropout, Lambda

print("Begin model construction")
stacks = []
conv_shapes = [1,3,5] # parallel filter sizes
depths = [6,8,8,6] # output depth at each convolutional layer

inputs = Input(shape=(160-CROP_SIZE,320,3))
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
model.fit_generator(train_generator, n_steps, epochs=1)

model.save('cropped_4convs_t2.h5')
print("Total Running Time: " + str((time.time() - base_time)//60) + "mins")





##
