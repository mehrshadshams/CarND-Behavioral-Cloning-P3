# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/nvidia_end2end_net.png "NVIDIA model"
[history]: ./images/history.png "Training history"
[left]: ./images/left.jpg "Left"
[center]: ./images/center.jpg "Center"
[right]: ./images/right.jpg "Right"
[translation]: ./images/center_shifted.jpg "center shift"
[brightness]: ./images/center_brightness.jpg "center brightness"
[shadow]: ./images/center_shadow.jpg "center shadow"
[flip]: ./images/center_flip.jpg "center flip"
[recovery1]: ./images/recovery1.jpg "Recovery 1"
[recovery2]: ./images/recovery2.jpg "Recovery 2"
[recovery3]: ./images/recovery3.jpg "Recovery 3"
[crop]: ./images/center_crop.jpg "Crop"
[activations]: ./images/activations.jpg "Activations"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I used the model proposed by NVIDIA in the paper **End to End Learning for Self-Driving Cars** (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) to train the car to drive. 
This model uses the structure described in the following picture:

![alt text][model]

The code below shows the output of the keras model. The model uses a normalization layer (implemented as a Lambda layer)
The simulator outputs images of size 320x160 pixels. I will go over the pre-processing of the images, but it's worth 
noting here that for the sake of making the training of the model faster, I resized the input images to 64x64
 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_3 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 5, 48)          43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 64)          27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 64)          36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 1, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               6500
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 143,419
Trainable params: 143,419
Non-trainable params: 0
_________________________________________________________________
None
```

The model has 3 convolution blocks of 5x5 with stride 2x2 and ELU activation followed by two
3x3 convolution blocks with stride 1x1 and ELU activation. The ELU activation is used because it helps with the learning 
speed. I added a drop out layer after the last convolution to combat over fitting. Then there is a Flatten layer and 3 
dense layers with activation ELU and finally a dense layer with only 1 neuron and no activation. This neuron predicts the 
steering wheel rotation angle.

I used mean squared error as the loss and I trained the model for 10 epochs using Adam optimizer with learning rate equal to 0.001 and I used learning rate decay 
equal to 0.95 
 
I used the training data captured from the first track. I combined driving in the center of the lane, recovering from the 
sides and also data captured focusing on the sharp turns. I will discuss the captured data in the next section.

After capturing data I split the data into training and validation datasets (10% validation)

```python
X_train, X_val, y_train, y_val = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.1)
```

Then I used a python generator to feed the data into the Keras model. I saved the model at the end of training
to be used in drive.py for testing. To make sure that I only save the model in epochs that actually lead to better validation accuracy I used
Keras checkpoints.

```python
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.5f}.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True)
```

This code makes sure that the model is saved only on epochs that actually resulted in lower validation loss. This way I can
easily pick the last best saved model for my testing. The following image shows the trainig/validation loss 
graph

![alt text][history]

## Acquiring data and augmentation

The simulator captures three sets of images. I used all three images for training. As part of the augmentation I randomly chose
between left and right images and applied a correction factor of 0.25 degree to the steering wheel angle captured for the center image

To make sure the resulting steering angle is between -1..1 I clipped the value.

![alt text][left]
![alt text][center]
![alt text][right]

The following code chooses one of the images in random:

```python
idx = np.random.randint(0, 3)
filename = files[idx]

if idx == 0: # LEFT
    angle += CORRECTION_FACTOR
elif idx == 2: # RIGHT
    angle -= CORRECTION_FACTOR

```

The following code shows the augmentation performed on the images

```python
def extend_image(image, angle):

    image, angle = translate_image(image, angle)
    image = add_shadow(image)
    image = augment_brightness(image)

    # Don't flip image if we're moving in straight line (almost)
    if np.random.uniform() > 0.5:
        image = np.fliplr(image)
        angle *= -1

    return image, angle
```

First I apply random translation of maximum 50 pixels. Then I added random
overlay mask to mimick shadows (for generalization of the model) The I added some brightness
adjustment and finally with a 50% chance I flipped the image because the track 1 has a high bias toward turn left
 
|  Translation             | Brightness Adjustment   | 
| ------------------------ | ----------------------  | 
| ![alt text][translation] | ![alt text][brightness] | 

|  Random shadow           | Flip   | 
| ------------------------ | ----------------------  | 
| ![alt text][shadow]      |  ![alt text][flip]      |

I also collected few images for recovering from the sides and going through sharp turns

|                        |                        |                        |
| ---------------------- | ---------------------  | ---------------------- |
| ![alt text][recovery1] | ![alt text][recovery2] | ![alt text][recovery3] | 

The last step of pre-processing was to crop the images to only limit to the road and remove
the hood and the sky and normalize the values between -0.5 and 0.5 which was done in the first layer (Lambda) of the model

![alt text][crop]

To run the model:

```bash
python drive.py model.h5
```

Finally here's the result of the model driving:

![alt text][result]

I was curious how will the activations look like, so I wrote the code in show_activations.py that computes the activations of all 
convolution layers and create a single image for each frame

<a href="http://www.youtube.com/watch?feature=player_embedded&v=_YpAa8rnSSw
" target="_blank"><img src="https://img.youtube.com/vi/_YpAa8rnSSw/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

and here's the activations for the recorded test

![alt text][activations_video]