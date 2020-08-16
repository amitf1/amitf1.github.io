---
title: "Bird Species Image Classification"
date: 2020-08-02
tags: [Computer Vision, Deep Learning]
header:
  image: "/images/Bird Species/birds4_1.jpg"
excerpt: "Computer Vision, Deep Learning"
mathjax: "true"
---


## Classify each image of a bird to one of 200 bird species, using Deep Leaning and Transfer Learning

The main goal of this project was to classify images of birds to their correct bird species.
We also built an API, where a user could load an image, or use the camera to take a picture, and get the classification of the bird and it's Wikipedia page.

### The Data
The data consists of about 30,000 images of birds with their species as labels.
The images' shape is 224x224x3.
Each class has around 140 train images.
This task is challenging because of several reasons:
- We have 200 classes, that's a lot, so there is definitely a room for mistakes.
- Some species looks similar to other species, where even human being, a domain expert could be wrong.
- Species contain images of male and female birds, which often look very differently, which makes it harder to group them together.
- Image backgrounds can harm generalization, if one bird is usually shows on snowy background, the model can fail when classifying this bird on other backgrounds.


<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/029.jpg" alt="Bird"> <img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/108.jpg" alt="Bird">
<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/111.jpg" alt="Bird"> <img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/018.jpg" alt="Bird">
### Data Preprocessing
We used ImageDataGenerator from Keras, to create a generator of batches of images, and rescale them by factor of 1/255.
```python
from keras.preprocessing.image import ImageDataGenerator

General_datagen = ImageDataGenerator(rescale=1./255)

train_data = General_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))
validation_data = General_datagen.flow_from_directory(VALIDATION_DIR, target_size=(224,224))
test_data = General_datagen.flow_from_directory(TEST_DIR, target_size=(224,224))
```

### Neural Network Architecture based on MobileNet

This architecture was built on top of **Mobilenet**, which was already trained on **Imagenet** and **freezing** it's weights.
We **throw away the top layer of mobilenet**, which is optimized to different number of classes, and also we don't want to predict at that level yet.
We **flatten** the results we get from mobilenet, and use relu for the non linearity.
Then a **dropout** layer - which we helped us overcome a minor overfitting issue.
Finally we have a **dense** layer with 200 outputs, 1 for each specie, with softmax so we get probabilities.

Only our layers were trained, **no fine-tuning was needed**

The nice thing about [MobileNet](https://arxiv.org/abs/1704.04861) is the use of Depthwise Separable Convolution which enables the network to be light, yet effective. You can read about it also here [MobileNet](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.applications import MobileNet

#Get Mobilenet pre-trained on Imagenet, with it's optimized weights, use it as a base for our model.
#Remove the classification top.
base_mobilenet = MobileNet(weights = 'imagenet', include_top = False,
                           input_shape = SHAPE)
base_mobilenet.trainable = False # Freeze mobilenet's weights.

model = Sequential()
model.add(base_mobilenet)
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(200))
model.add(Activation('softmax'))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.05),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/nn.png" alt="Neural Network Architecture">

### Training
We combined the following callbacks:
* Reduce learning rate on Plateau was used as our learning rate scheduler reduces the learning rate by factor of 0.5 when no improvement is made in the validation loss for 2 epochs.
* Early Stopping - stopping when we don't improve for 5 epochs.

### Results
After trying different models and architectures, we managed to get satisfying results where the model shown above was our winner.
We got around **94% average F1 score** on the validation and test sets.
(The confusion matrix and the classification report are too big to fit here, but you can see them on the notebook that is in the repository)

#### Accuracy Over Epochs

<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/acc.png" alt="Accuracy">

#### Loss Over Epochs
<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/loss.png" alt="Loss">

### API
This is a flask app, currently locally hosted.
The user, as shown below, can choose to upload an image or to take a picture using the camera.
The images below show the demo of taking a picture using the camera.
We present to our camera a mobile phone with a bird image from Google, we got the correct classification and the corresponding Wikipedia page was opened in a new tab.
Even with the poor image quality we took the model classified correctly.
<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/API11.png" alt="API">

<img src="{{ site.url }}{{ site.baseurl }}/images/Bird Species/API.png" alt="API">

[Repository with the Full Code](https://github.com/amitf1/Birds_Classifier)
