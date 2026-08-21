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

## Dataset and IMU Input

The system uses 6-axis inertial measurement unit (IMU) data.

### Input Channels

| Channel | Signal |
|---|---|
| aX | Accelerometer X-axis |
| aY | Accelerometer Y-axis |
| aZ | Accelerometer Z-axis |
| gX | Gyroscope X-axis |
| gY | Gyroscope Y-axis |
| gZ | Gyroscope Z-axis |


### Window Configuration

Each input sample is segmented into fixed-size windows:

```text
Window size:

128 samples × 6 channels

Total raw input values:

768
```

## Gesture Classification

The system performs four-class gesture recognition.

| Class ID | Gesture |
|---|---|
| 0 | circle |
| 1 | left_right |
| 2 | rest |
| 3 | up_down |


Each gesture class is evaluated during training, validation, and testing.

# Study 1 — Neural Network Compression Benchmark

## Objective

The first study evaluates different neural network architectures for embedded deployment.

The goal is to reduce:

- Model size
- Computational cost
- Memory usage
- Inference latency

while maintaining classification accuracy.


## Compared Architectures

The following models are evaluated:

- Standard CNN
- Depthwise CNN
- INT8 quantized models

## Study 1 Results

### INT8 Model Comparison

All CNN models were converted to TensorFlow Lite INT8 format for embedded deployment.

| Model | Parameters | NN MACs |
|---|---:|---:|
| Standard CNN INT8 | 2,660 | 160,320 |
| Depthwise CNN INT8 | 1,330 | 52,544 |


The Depthwise CNN reduces the computational cost compared with the Standard CNN while maintaining the same classification performance.

## Neural Network Computational Cost

The neural-network computational cost is measured using Multiply-Accumulate operations (MACs).

![Neural Network MAC Comparison](Figure/final_nn_mac_comparison.png)


The comparison shows that depthwise separable convolutions significantly reduce the number of operations required for inference.

## Embedded Arduino Benchmark

The final INT8 models were evaluated on the Arduino Nano 33 BLE.

### Standard CNN INT8

| Metric | Value |
|---|---:|
| Flash Memory | 194,408 bytes |
| Tensor Arena | 6,644 bytes |
| Inference Time | 25.370 ms |
| Total Processing Time | 26.760 ms |


### Depthwise CNN INT8

| Metric | Value |
|---|---:|
| Flash Memory | 205,424 bytes |
| Tensor Arena | 7,156 bytes |
| Inference Time | 16.880 ms |
| Total Processing Time | 18.330 ms |

## Arduino Inference Latency Comparison

![Arduino Inference Latency](Figure/final_arduino_inference_latency.png)


Depthwise CNN provides lower inference latency compared with the Standard CNN, making it more suitable for embedded real-time applications.


