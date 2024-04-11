# Radar Triggered Object Classification
TensorFlow Lite Radar/IR triggered webcam visual object classification

## Purpose
This is something I'm using on a device where power is limited and so uses a sensor to trigger the capture and recognition of what is seen by the web camera attached to the system.
This could be a Radar or IR sensor, in my use case this is a radar sensor attached GPIO 4 on a raspi 3b+.  Adjust this to your need.

## Requirements
Tensorflow Lite and YoloV5 model - sure this could be many different models that would be better! OpenCV for the capture from the webcam device.
A sensor with its detection pin on GPIO4, a usb webcam
