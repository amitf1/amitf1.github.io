---
title: "Time Series Prediction & Anomaly Detection"
date: 2020-08-02
tags: [Time Series, Deep Learning, Anomaly Detction]
header:
  image: "/images/time-series/time-series.jpg"
excerpt: "Time Series, Deep Learning, Anomaly Detction"
mathjax: "true"
---


## Predictiong Iot Sensors' Values and Detecting Anomalies, Using Neural Networks

This project was done as part of my Internship as a Data Scientist.
The main goal was to improve a simplier model that predicts values of sensors to set a baseline for real time anomaly detection.
The motivation for anomaly detction is to find pollution events.

My model was designed to find patterns and aim to predict a **normal** behavior of the data.
After achieving goo results on the validation and test set, the model is "trusted" as a baseline for normal behavior, which then if a significant deviation is detected, we can mark it as an anomaly.

The specifics are confidential so this post will show my work without the data, and censored findings.

### Data Preprocessing

The preprocessing was pretty simple, standardization, removing outliers, and filling missing values by interpolating and padding.

### Creating Time Windows

For my neural net's design, my data should consist of windows of two days - 192 samples and labels of one day - 96 samples.
Then for fitting into the first layer of the network, which is a Convolution layer (elaborated next) the windows should be reshaped as a matrix, the two days windows are split to 8 subsequences, that way we can see patterns also in smaller resolutions (2 days ==>> quarters of days).



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

You can also put it inline $$z=x+y$$
