# **Behavioral Cloning**

## Satchel Grant

### A self driving virtual car

---

Links to [track1](./track1.mp4) and [track2](./track2.mp4) video files of the car driving itself.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the Udacity car simulator to collect driving data
* Build a convolutional neural network in Keras to predict steering angles from images
* Train and validate the model with the driving data
* Successfully train car to drive safely around a track


[//]: # (Image References)

[image1]: ./examples/centered_cropped.jpg "Cropped"
[image2]: ./examples/centered.jpg "Centered"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/recovery4.jpg "Recovery Image"
[image7]: ./examples/canny_edges.png "Canny Edges"

---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

* Note: The simulator contains a small bug that sometimes pauses the car. If this happens, simply press the back arrow or 's' key on your keyboard to release the pause.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. It contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is a convolutional neural network that begins with a preprocessing step to center and normalize the inputs (model.py line 99), then followed by 4 sets of a 1x1, 3x3, and 5x5 filter size convolutions. Each convolutional set is followed by a max pooling layer (model.py lines 103-116). There is then a dropout layer (model.py line 119) followed by 2 fully connected layers of sizes 100 and 25, and finally an output layer (model.py lines 122-124).

At each convolutional layer the model splits and runs a 1x1, a 3x3, and a 5x5 filter on the incoming activation inputs. It is important to note that the filters are not run sequentially but rather in parallel to each other. The padding is 'SAME' so that the outputs from each filter is of the same size. These outputs are then stacked and used as the activations for the next layer. This model architecture is inspired by the Inception Net, but simpler.

The model includes an ELU activation at each convolution and each fully connected layer to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer, as described in the previous section, in order to reduce overfitting (model.py line 118).

The model was trained and validated on multiple datasets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

#### 4. Appropriate training data

Training data was carefully curated to keep the vehicle driving on the road. The bulk of the training data centered the vehicle in the lane. There was also a dataset that had the vehicle recovering from the left and right sides of the road, and finally there was a dataset that focused on the curves of the road.

For details about how I created the training data, see the next section.

---
## Model and Training Strategy

#### 1. Solution Design Approach

The strategy for designing a predictive model was to start simple and progressively add convolutional and fully connected layers as needed.

My first step was to use a convolution neural network model that derives its architecture from both the LeNet and Inception Net architectures. I thought this model might be appropriate because I have had success using it in other image classification projects, it is quick to train with relatively few parameters, and it is easy to adjust and augment.

In order to train and gauge how well the model was working, I split my image and steering angle data into a training and validation set. I cropped the top of images off for a smaller image size so that the model would train and validate quicker. I found that my model had a higher mean squared error on the training set than on the validation set. This implied that the model was not over fitting with its single dropout layer.

I tested the model on the simulator to see how well the car could drive around track one. The vehicle fell off the track at a curve in which the outer edge leads into a flat dirt area that looks similar to a road. It also failed to make it around a sharper turn later in the track. To fix this, I added another convolutional layer and another fully connected layer to my model. This seemed to improve the car's overall driving ability, but it still fell off the road at the same spots. I tried adding another convolutional layer but it took longer to train and didn't seem to improve results.

I then tried redoing the curves and corrections driving data. This drastically helped the car's ability to drive around the curves, but it still struggled distinguishing the dirt area from the road.

At this point, I decided that augmenting the images to indicate the difference between road and dirt would likely give the quickest improvements. I added a canny edge detection preprocessing step (canny_augment.py) to each of the data images and added it to the autonomous driving pipeline (drive.py line 19,67). This drastically improved performance and the car successfully drove around track one.

I followed nearly the same process for training the model to navigate track 2. I included, however, the data from track one in the training for track 2. This resulted in a robust model that successfully navigates both tracks forward and backward.

#### 2. Final Model Architecture

The final model architecture (model.py lines 92-124) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 110x320x3 image   							|
| Preprocessing         		| Centers and normalizes image pixels   							|
| Convolution 1x1, 3x3, 5x5     	| 1x1 stride, same padding, depth 6, outputs 110x320x18 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 55x320x18 				|
| Convolution 1x1, 3x3, 5x5     	| 1x1 stride, same padding, depth 8, outputs 55x160x24 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 27x80x24 				|
| Convolution 1x1, 3x3, 5x5     	| 1x1 stride, same padding, depth 8, outputs 27x80x24 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 13x40x24 				|
| Convolution 1x1, 3x3, 5x5     	| 1x1 stride, same padding, depth 6, outputs 13x40x18 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 6x20x18 				|
| Dropout	      	| 0.5 probability 				|
| Fully connected		| 2160x100, outputs x100        									|
| ELU					|												|
| Fully connected		| 100x25, outputs x25        									|
| ELU					|												|
| Fully connected		| 25x1, outputs x1        									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps going forward on track one and two laps going the reverse direction around track one, keeping to the center of the lane as best as possible. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering to the center of the road from the left side and right edges of the road so that the vehicle would learn to correct itself if it were to get off track. These images show the progression of a recovery :

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

And finally I recorded the vehicle focussing on good driving around the curves only.

To augment the data sat, I flipped a portion of the images (and corresponding angles) in an effort to double the data and create a more generalized model.

After the collection and augmentation, I had about 22,302 images. I preprocessed them by first cropping off the top 50 pixels of the images to allow the model to focus better on the road and to cut RAM use and speed up training.

![alt text][image2]
![alt text][image1]

And then I added canny edge detection to the images with a lower threshold of 30 and an upper of 160. I considered further manipulation of the canny edges to focus on isolation of the edges of the road, but the classification model was able to ignore unimportant canny edges on its own. In the end, I left the canny edges as they came. I believe they highlighted both the texture of the road and the edges of the road making these features easier for the model to notice.

![alt text][image7]

I finally shuffled the images into a random order and put 25% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I approached the model training by running bursts of epochs using the weights from the last training session often changing the data. Overall, the total epoch count was likely in the 40-50 range. I used an adam optimizer so that manually training the learning rate wasn't necessary.
