# Keenmind

## Overview

Keenmind aims to provide dnd players a way to track dice rolls whilst retaining the fun of rolling dice. Keenmind utilises state of the art object detection methods to automatically detect what dice were rolled, and what their values were.
Early Work

This repo is not the first of its type. Previous work has proved that dice detection is possible using the YoloV3 architecture. Unfortunately, this work came to halt due to a lack of data. However, a test bench is in the works that will enable faster data collection and labelling.

## ML Architecture

Keenmind is split into two ML components:

1. Object Detection: The first model is used to isolate the dice types and locations. This uses the YoloV3 architecture trained on a custom dataset (this dataset will be published elsewhere at a later date).

2. Digit Classification: This is the hello world of the deep learning computer vision, usually using the MNIST dataset. Therefore, it is possible to utilise transfer learning here, although the data distrubution is so simple, the gains from this may be negligble.
