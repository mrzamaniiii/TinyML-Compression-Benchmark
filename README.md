# TinyML Compression Benchmark for IMU Gesture Recognition

## Overview

This project investigates efficient TinyML solutions for real-time IMU-based gesture recognition on the Arduino Nano 33 BLE.

The main objective is to reduce computational cost, memory usage, and inference latency while maintaining high classification accuracy.

Two optimization approaches are investigated:

1. Neural network compression:
   - Standard CNN
   - Depthwise CNN
   - INT8 quantization

2. Feature engineering:
   - RMS statistical features
   - Lightweight dense neural network

The final models are deployed and benchmarked on a real embedded platform using TensorFlow Lite Micro.

## System Pipeline

```text
6-Axis IMU Sensor
        |
        v
Window Segmentation
(128 × 6 samples)
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
```

## Hardware and Software

### Target Hardware

- Arduino Nano 33 BLE

### Software Framework

- TensorFlow Lite Micro
- TensorFlow Lite INT8 Quantization
- Embedded C/C++ model deployment

### Deployment Goal

The objective is to achieve real-time gesture recognition directly on a microcontroller without cloud processing.

The objective is to achieve real-time gesture recognition directly on a microcontroller without cloud processing.
