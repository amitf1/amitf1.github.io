---
title: "Time Series Prediction & Anomaly Detection"
date: 2020-08-02
tags: [Time Series, Deep Learning, Anomaly Detction]
header:
  image: "/images/time-series/time-series.jpg"
excerpt: "Time Series, Deep Learning, Anomaly Detction"
mathjax: "true"
---


## Predictiong Iot Sensors' Values and Detecting Anomalies, Using Deep Learning

The main goal of this project was to achieve better results than a simplier existing model, that predicts values of sensors to set a baseline for real time anomaly detection.
The motivation for anomaly detction is to find harmful events.

My model was designed to find patterns and aim to predict a **normal** behavior of the data.
After achieving satisfying results on the validation and test set, the model is "trusted" as a baseline for normal behavior, which then if a significant deviation is detected, we can mark it as an anomaly.

The specifics are confidential so this post will show my work without the data, and censored findings.

### Data Preprocessing

The preprocessing was pretty simple:
* Standardization
* Removing outliers
* Filling missing values by interpolating and padding

### Creating Time Windows

For my neural net's design, my data should consist of windows of two days - 192 samples and labels of one day - 96 samples.
Then for fitting into the first layer of the network, which is a Convolution layer (elaborated next) the windows should be reshaped as a matrix, the two days windows are split to 8 subsequences, that way we can see patterns also in smaller resolutions (2 days ==>> quarters of days).

### Neural Network Architecture CNN + Stacked LSTM's

This architecture consists of a one  dimensional Convolution layer, followed by Max Pooling,
then the output is flattened to fit the next layer, which is a LSTM layer, then another layer of LSTM, followed by a dense layer that outputs our desired 96 values.
Huber loss is used, as it is better for dealing with outliers, MAE is also tracked along training.

I used tensorflow.keras for creating the network:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D, Dense, LSTM, TimeDistributed, Conv1D, Flatten

time_steps_per_seq = int(CNFG.HIST_SIZE // CNFG.SUBSEQ_N)
        multi_step_model = Sequential(
            [
                TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                input_shape=(CNFG.SUBSEQ_N, time_steps_per_seq, 1)),
                TimeDistributed(MaxPooling1D(pool_size=2)),
                TimeDistributed(Flatten()),
                LSTM(32, return_sequences=True),
                LSTM(16, activation='relu'),
                Dense(CNFG.TARGET_SIZE)
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        loss = tf.keras.losses.Huber()
        multi_step_model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
```

<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/nn.png" alt="Neural Network Architecture">

### Training
For the research part, I combined the callbacks - Reduce learning rate on Plateau, which is a nice learning rate scheduler that reduces the learning rate when no improvement is made, and also Early Stopping.
For deployment I found the learning rate of 3e-4 was stable enough and achieves good results.
### Censored findings
Results were improved significantly, as MAPE is around 7%.
The model was able to generalize well, even when training on one location and predicting on another location achieves better results then the original model.
The model can handle a gap between training and prediction.
* The y axis could not be revealed

#### Predicted values against Ground Truth
The blue plot is the predicted values, the red one is the ground truth.
We can see how close are the predicted values to the actual values.
<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/pred.png" alt="Prediction">

#### Anomaly Detection
We can see how on most of the day the predicted values are close to the actual values, while the spikes, which is above the set threshold are marked as anomalies.
<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/anomaly.png" alt="Anomaly Detection">

#### Generalization capabilities
We can see how the predicted values are still pretty accurate, even when the model is trained on data take from different location.
<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/generalization.png" alt="Generalization">

#### Comparing to Facebook's Prophet
We can see how prophet's prediction is worse then the above results.
Note that this was trained on the same location of prediction, and predicted without a gap from the training time, while the above results were after the model was trained on a different location and with a 1 month gap between training and prediction.
<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/prophet.png" alt="Prophet">
[Repository with the Full Code](https://github.com/amitf1/Conv_LSTM_Time_Series_Prediction)
<!--

What about a [link](https://github.com/amitf1)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$ -->
