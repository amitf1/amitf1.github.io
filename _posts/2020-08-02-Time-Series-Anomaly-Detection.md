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

This project was done as part of my Internship as a Data Scientist.
The main goal was to improve a simplier model that predicts values of sensors to set a baseline for real time anomaly detection.
The motivation for anomaly detction is to find pollution events.

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

This architecture consists of:
a one  dimensional Convolution layer, followed by Max Pooling,
then the output is flattened to fit the next layer, which is a LSTM layer, then another layer of LSTM, followed by a dense layer that outputs our desired 96 values.

```python

multi_step_model = Sequential(
[
  TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'),
                  input_shape=X_train.shape[1:]),
  TimeDistributed(MaxPooling1D(pool_size=2)),
  TimeDistributed(Flatten()),
  LSTM(32, return_sequences=True),
  LSTM(16, activation='relu'),
  Dense(96)
]
)
multi_step_model.compile(optimizer='adam', loss=Huber(), metrics=["mae"])
```

<img src="{{ site.url }}{{ site.baseurl }}/images/time-series/nn.png" alt="Neural Network Architecture">
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
