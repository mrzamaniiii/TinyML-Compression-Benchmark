# TinyML Gesture Classification on Arduino Nano 33 BLE

Real-time IMU gesture classification using an INT8 Depthwise CNN deployed with TensorFlow Lite for Microcontrollers.

This project covers the complete TinyML deployment pipeline, including IMU data acquisition, neural-network training and quantization, embedded preprocessing, TensorFlow Lite Micro integration, memory debugging, model alignment, and on-device inference.

---

## Project Overview

The goal of this project is to deploy a lightweight gesture-classification neural network directly on an Arduino Nano 33 BLE platform and ultimately perform gesture recognition entirely on-device.

The final runtime pipeline is:

```text
On-board IMU
     ↓
6-axis IMU data
ax, ay, az, gx, gy, gz
     ↓
128 × 6 sample window
     ↓
Standardization
raw_mean / raw_std
     ↓
INT8 quantization
     ↓
Depthwise CNN
     ↓
TensorFlow Lite Micro
     ↓
4-class output
     ↓
Gesture + confidence
