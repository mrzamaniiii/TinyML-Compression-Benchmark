TinyML Compression Benchmark for IMU Gesture Recognition

Overview

This project investigates efficient TinyML solutions for IMU-based
gesture recognition on Arduino Nano 33 BLE.

The objective is reducing computational cost and memory usage while
maintaining high classification accuracy.

System Pipeline

IMU Sensor → Window Segmentation → Feature Extraction / CNN → INT8
TinyML Model → Arduino Nano 33 BLE → Gesture Prediction

Hardware

Target: - Arduino Nano 33 BLE

Framework: - TensorFlow Lite Micro - INT8 quantization - Embedded C/C++
model deployment

Dataset

6-axis IMU input:

aX

aY

aZ

gX

gY

gZ

Window: 128 × 6 samples

Classes:

Class   Gesture

0       circle
1       left_right
2       rest
3       up_down

Study 1 --- CNN Compression

Model                  Parameters     MACs

Standard CNN INT8            2660   160320
Depthwise CNN INT8           1330    52544





Study 2 --- Feature Engineering

All representations use the same Dense classifier.

Input → Dense(64) → Dense(32) → Dense(4)

Representation     Features   Accuracy

Raw Signal              768       100%
RMS                       6       100%
FFT                      48       100%
Spectral                 12    96.875%



Final Model --- RMS Tiny INT8

Architecture:

6 RMS Features

↓

Dense(16)

↓

Dense(8)

↓

Softmax(4)

Performance:

Metric                   Value

Parameters                 284
NN MACs                    256
TFLite Size           3.414 KB
Accuracy                  100%
Macro F1                   1.0
Arduino Inference     0.165 ms
Total Processing      0.346 ms

Deployment Files

rms_tiny_model.h

rms_feature_scaler.h

depthwise_cnn_model.h

depthwise_cnn_scaler.h

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

The RMS Tiny INT8 model achieves the same classification accuracy as
larger CNN models while drastically reducing computational complexity,
making it suitable for real-time embedded TinyML deployment.
