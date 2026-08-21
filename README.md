TinyML Compression Benchmark for IMU Gesture Recognition

Overview

This project explores efficient TinyML solutions for real-time IMU
gesture recognition on the Arduino Nano 33 BLE.

The goal is to reduce computational cost, memory usage, and inference
latency while maintaining high classification accuracy.

Two optimization approaches are investigated:

Neural network compression:

Standard CNN

Depthwise CNN

INT8 quantization

Feature engineering:

RMS statistical features

Lightweight dense neural network

System Pipeline

6-Axis IMU Sensor
        |
        v
Window Segmentation
(128 x 6 samples)
        |
        v
Feature Extraction / CNN Processing
        |
        v
INT8 TinyML Model
        |
        v
Arduino Nano 33 BLE
        |
        v
Gesture Prediction

Hardware and Software

Target hardware:

Arduino Nano 33 BLE

Software:

TensorFlow Lite Micro

TensorFlow Lite INT8 quantization

Embedded C/C++ model headers

Dataset

Input channels:

Channel   Signal

aX        Accelerometer X
aY        Accelerometer Y
aZ        Accelerometer Z
gX        Gyroscope X
gY        Gyroscope Y
gZ        Gyroscope Z

Window size:

128 samples x 6 channels

Gesture classes:

Class   Gesture

0       circle
1       left_right
2       rest
3       up_down

Study 1 --- CNN Compression Benchmark

The first study evaluates CNN architectures for embedded deployment.

Model Comparison

Model                  Parameters     MACs

Standard CNN INT8            2660   160320
Depthwise CNN INT8           1330    52544

Neural Network Computational Cost



Arduino Inference Latency



Study 2 --- Feature Engineering

All feature representations use the same classifier:

Input
 |
 v
Dense(64)
 |
 v
Dense(32)
 |
 v
Dense(4)

Representation Comparison

Representation     Input Features   Accuracy

Raw Signal                    768       100%
RMS                             6       100%
FFT                            48       100%
Spectral                       12    96.875%

Accuracy vs Model Complexity



Final Selected Model --- RMS Tiny INT8

The final embedded solution uses RMS feature extraction followed by a
tiny neural network.

Architecture:

6 RMS Features
        |
        v
Dense(16)
        |
        v
Dense(8)
        |
        v
Softmax(4)

Embedded Performance

Metric                   Value

Parameters                 284
NN MACs                    256
TFLite Size           3.414 KB
Accuracy                  100%
Macro F1                1.0000
Arduino Inference     0.165 ms
Total Processing      0.346 ms

Deployment Files

Generated files:

rms_tiny_model.h
rms_feature_scaler.h
depthwise_cnn_model.h
depthwise_cnn_scaler.h

These files can be integrated into TensorFlow Lite Micro Arduino
projects.

Results Summary

The RMS Tiny INT8 model achieves:

Same accuracy as larger CNN models

256 neural-network MAC operations

Very small TFLite model size

Real-time Arduino inference

Repository Structure

TinyML-Compression-Benchmark/

├── Arduino/
├── Code/
├── Data/
├── Figure/
├── Results/
├── Presentation/
└── README.md

Conclusion

Feature engineering combined with a lightweight neural network provides
an efficient TinyML solution for embedded gesture recognition.

The final RMS Tiny INT8 model achieves high accuracy with significantly
reduced computational requirements, making it suitable for real-time
microcontroller deployment.
