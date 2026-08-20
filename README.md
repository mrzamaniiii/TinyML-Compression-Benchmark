# TinyML Compression & Feature Engineering for Gesture Recognition

<p align="center">
<img src="Figure/final_arduino_inference_latency.png" width="750">
</p>

<p align="center">
<b>Real-time IMU gesture recognition on Arduino Nano 33 BLE using TinyML optimization, INT8 quantization, CNN compression, and RMS feature engineering.</b>
</p>


---

# Overview

This project investigates efficient TinyML approaches for embedded gesture recognition using a 6-axis IMU sensor and an Arduino Nano 33 BLE platform.

The main objective is to understand how embedded machine-learning systems can be optimized through:

- Neural-network compression
- INT8 quantization
- Depthwise separable convolutions
- Classical signal-processing feature extraction
- TensorFlow Lite Micro deployment
- Physical Arduino benchmarking


The project addresses two main research questions:

1. **How much can a TinyML neural network be compressed while maintaining classification accuracy?**

2. **Can signal-processing-based feature extraction outperform neural-network-only compression for extremely small embedded systems?**


---

# Table of Contents

- [System Overview](#system-overview)
- [Dataset](#dataset)
- [Study 1 - TinyML Compression Benchmark](#study-1---tinymL-compression-benchmark)
- [Study 2 - Feature Engineering Study](#study-2---feature-engineering-study)
- [Final Embedded Comparison](#final-embedded-comparison)
- [Arduino Deployment](#arduino-deployment)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [Conclusion](#conclusion)


---

# System Overview

The complete embedded pipeline is:
