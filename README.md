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
