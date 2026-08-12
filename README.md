# TinyML Compression & Feature Engineering for Gesture Recognition

Real-time IMU gesture classification and embedded TinyML optimization on the **Arduino Nano 33 BLE Sense**.

This project investigates two complementary research questions:

1. **How much can a TinyML model be compressed while preserving classification performance?**
2. **Is classical signal-processing-based feature extraction still valuable in modern TinyML systems?**

The project combines neural-network compression, INT8 quantization, efficient convolutional architectures, handcrafted signal features, TensorFlow Lite for Microcontrollers, and physical Arduino benchmarking.

---

# Research Objectives

The work is organized into two main experimental studies.

## Study 1 — TinyML Compression Benchmark

### Research Question

> For an embedded TinyML gesture-classification task, which optimization technique provides the greatest reduction in memory usage and execution cost with the smallest loss in classification accuracy?

The following model families are compared:

```text
Dense Network
      ↓
Standard 1D CNN
      ↓
Quantized CNN
      ↓
Depthwise Separable CNN
      ↓
Quantized Depthwise CNN
```

The models are evaluated using:

```text
Accuracy
Model Size
Parameter Count
MAC Count
Flash Usage
RAM Usage
Tensor Arena Usage
Inference Time
```

The final comparison will use measurements obtained directly from the Arduino platform.

---

## Study 2 — TinyML Feature Engineering Study

### Research Question

> Is handcrafted signal feature extraction still useful for TinyML when compared with learning directly from raw sensor signals?

The same gesture-classification problem is studied using different signal representations:

```text
Raw IMU Signal
      vs
RMS Features
      vs
FFT Features
      vs
Spectral Features
```

A feature-engineered neural classifier has also been developed as an advanced resource-efficient candidate.

The feature study evaluates:

```text
Classification accuracy
Number of input features
Model parameters
Model size
Feature-extraction latency
Classifier latency
Total pipeline latency
Flash usage
RAM usage
```

This is particularly important because a small neural classifier does not necessarily imply a low-cost embedded pipeline: feature extraction also consumes computation and memory.

---

# System Overview

The complete embedded pipeline is:

```text
Physical Gesture
      ↓
6-Axis IMU
      ↓
128 × 6 Sensor Window
      ↓
Signal Preprocessing
      ↓
Raw Model
      OR
Feature Extraction
      ↓
TinyML Classifier
      ↓
INT8 Inference
      ↓
Gesture Prediction
      ↓
Confidence Score
```

The system is designed to operate fully on-device without cloud inference or an external computer.

---

# Gesture Classification Task

The system uses six-axis inertial measurements:

```text
ax
ay
az
gx
gy
gz
```

Each input window contains:

```text
128 samples × 6 channels
```

corresponding to:

```text
768 raw sensor values
```

Four gesture classes are considered:

```text
0 → circle
1 → left_right
2 → rest
3 → up_down
```

The same class ordering is maintained throughout:

```text
Dataset generation
Training
Evaluation
TFLite conversion
Arduino deployment
```

---

# Dataset

The current dataset contains:

```text
320 total windows
```

with:

```text
80 samples per gesture class
```

The main model-compression benchmark uses a fixed:

```text
80% training
10% validation
10% testing
```

split:

```text
Training samples:      256
Validation samples:     32
Test samples:           32
```

Input shape:

```text
128 × 6
```

The test set contains eight samples from each gesture class.

Normalization statistics are calculated using the **training set only** to prevent validation or test information from leaking into the training pipeline.

---

# Study 1 — TinyML Compression Benchmark

Seven configurations have currently been evaluated during development.

The five core architectures relevant to the compression study are:

```text
Model 1 — Dense Raw FP32
Model 2 — Standard CNN FP32
Model 3 — Standard CNN INT8
Model 4 — Depthwise CNN FP32
Model 5 — Depthwise CNN INT8
```

Two additional feature-based models belong primarily to the Feature Engineering study:

```text
Model 6 — Feature Dense FP32
Model 7 — Feature Dense INT8
```

---

# Model 1 — Dense Raw FP32

The baseline classifier operates directly on the flattened raw IMU window.

Architecture:

```text
128 × 6
   ↓
Flatten
   ↓
768
   ↓
Dense 64
   ↓
Dense 32
   ↓
Dense 4
```

Results:

```text
Test accuracy:       100.00%
Parameters:          51,428
TFLite FP32 size:    203.93 KB
Average confidence:  0.999924
```

The Dense model provides a useful accuracy baseline but requires significantly more parameters and storage than the optimized architectures.

---

# Model 2 — Standard 1D CNN FP32

Architecture:

```text
128 × 6
   ↓
Conv1D
   ↓
MaxPooling1D
   ↓
Conv1D
   ↓
MaxPooling1D
   ↓
GlobalAveragePooling1D
   ↓
Dense 16
   ↓
Dense 4
```

Results:

```text
Test accuracy:       100.00%
Parameters:          2,660
Approx. MACs:        160,320
TFLite FP32 size:    15.70 KB
Average confidence:  0.999818
Host latency:        0.0147 ms
```

Compared with the Dense baseline, the Standard CNN substantially reduces the number of parameters and model size while preserving classification performance.

---

# Model 3 — Standard CNN Full INT8

The Standard CNN was converted to a fully INT8-quantized TensorFlow Lite model.

Results:

```text
Float accuracy:      100.00%
INT8 accuracy:       100.00%
Accuracy drop:       0.00 percentage points

Parameters:          2,660
Approx. MACs:        160,320
TFLite INT8 size:    10.71 KB
Average confidence:  0.996094
Host mean latency:   0.0173 ms
```

Model-size reduction relative to the FP32 Standard CNN:

```text
31.78%
```

INT8 quantization therefore reduced model storage without reducing accuracy on the current test set.

---

# Model 4 — Depthwise Separable CNN FP32

Architecture:

```text
128 × 6
   ↓
SeparableConv1D
   ↓
MaxPooling1D
   ↓
SeparableConv1D
   ↓
MaxPooling1D
   ↓
GlobalAveragePooling1D
   ↓
Dense 16
   ↓
Dense 4
```

Results:

```text
Test accuracy:       100.00%
Parameters:          1,330
Approx. MACs:        52,544
TFLite FP32 size:    11.94 KB
Average confidence:  0.999713
Host latency:        0.0107 ms
```

Compared with the Standard CNN:

```text
Parameter reduction:    50.00%
MAC reduction:          67.23%
Model-size reduction:   23.96%
```

with no reduction in test accuracy.

---

# Model 5 — Depthwise CNN Full INT8

Results:

```text
Float accuracy:       100.00%
INT8 accuracy:        100.00%
Accuracy drop:        0.00 percentage points

Parameters:           1,330
Approx. MACs:         52,544
TFLite INT8 size:     11.30 KB
Average confidence:   0.996094
Host mean latency:    0.0150 ms
```

INT8 model-size reduction compared with the FP32 Depthwise model:

```text
5.32%
```

This architecture has also been successfully executed using TensorFlow Lite Micro on the physical Arduino.

Previously measured on-device inference latency:

```text
~20.2 ms
```

Host and microcontroller latency measurements are not directly comparable.

---

# Study 1 — Current Compression Results

| Model              | Accuracy | Parameters | Approx. MACs | TFLite Size |
| ------------------ | -------: | ---------: | -----------: | ----------: |
| Dense Raw FP32     |  100.00% |     51,428 |          N/A |   203.93 KB |
| Standard CNN FP32  |  100.00% |      2,660 |      160,320 |    15.70 KB |
| Standard CNN INT8  |  100.00% |      2,660 |      160,320 |    10.71 KB |
| Depthwise CNN FP32 |  100.00% |      1,330 |       52,544 |    11.94 KB |
| Depthwise CNN INT8 |  100.00% |      1,330 |       52,544 |    11.30 KB |

All five compression-study models currently achieve:

```text
100% test accuracy
```

on the fixed 32-window test set.

The physical Arduino benchmark is still required before selecting the best embedded architecture.

---

# Study 2 — TinyML Feature Engineering

The second part of the project investigates whether preprocessing the IMU signal using classical signal-processing techniques can reduce the complexity required from the neural network.

The target comparison is:

```text
Raw Signal
    vs
RMS
    vs
FFT
    vs
Spectral Features
```

This controlled experiment is still being finalized.

---

# Feature-Based Neural Classifier

A more advanced feature-engineered pipeline has already been developed.

Each raw input window:

```text
128 × 6
```

is transformed into:

```text
72 engineered features
```

before neural classification.

Classifier architecture:

```text
72
 ↓
Dense 16
 ↓
Dense 8
 ↓
Dense 4
```

Two versions have been evaluated:

```text
Feature Dense FP32
Feature Dense INT8
```

---

# Model 6 — Feature Dense FP32

Results:

```text
Test accuracy:             100.00%
Parameters:                1,340
Classifier MACs:           1,312
TFLite FP32 size:          7.45 KB
Average confidence:        0.999535

Host feature latency:      0.98795 ms
Host classifier latency:   0.00229 ms
Host total pipeline:       0.99024 ms
```

The reported MAC count represents only the neural classifier.

```text
Classifier MACs
≠
Complete pipeline computation
```

Feature-extraction computation must also be considered.

---

# Model 7 — Feature Dense Full INT8

Results:

```text
Float accuracy:            100.00%
INT8 accuracy:             100.00%
Accuracy drop:             0.00 percentage points

Parameters:                1,340
Classifier MACs:           1,312
TFLite INT8 size:          4.48 KB
Average confidence:        0.996094

Host feature latency:      1.04228 ms
Host classifier latency:   0.00261 ms
Host total pipeline:       1.04489 ms
```

Model-size reduction relative to Feature Dense FP32:

```text
39.81%
```

The INT8 confusion matrix remained perfect:

```text
[[8 0 0 0]
 [0 8 0 0]
 [0 0 8 0]
 [0 0 0 8]]
```

Precision, recall, and F1-score were:

```text
1.0000
```

for all four gesture classes on the current test set.

---

# Combined Model Benchmark

| Model                  |    Accuracy | Parameters | Approx. MACs | TFLite Size |
| ---------------------- | ----------: | ---------: | -----------: | ----------: |
| Dense Raw FP32         |     100.00% |     51,428 |          N/A |   203.93 KB |
| Standard CNN FP32      |     100.00% |      2,660 |      160,320 |    15.70 KB |
| Standard CNN INT8      |     100.00% |      2,660 |      160,320 |    10.71 KB |
| Depthwise CNN FP32     |     100.00% |      1,330 |       52,544 |    11.94 KB |
| Depthwise CNN INT8     |     100.00% |      1,330 |       52,544 |    11.30 KB |
| Feature Dense FP32     |     100.00% |      1,340 |       1,312* |     7.45 KB |
| **Feature Dense INT8** | **100.00%** |  **1,340** |   **1,312*** | **4.48 KB** |

* Feature-model MAC values include only the neural classifier and not feature-extraction computation.

---

# Maximum Model Compression Observed

The original Dense FP32 model requires:

```text
203.93 KB
```

The Feature Dense INT8 classifier requires:

```text
4.48 KB
```

This represents approximately:

```text
97.8% reduction in TFLite model size
```

while preserving:

```text
100% test accuracy
```

on the current test dataset.

Trainable parameters were also reduced from:

```text
51,428
```

to:

```text
1,340
```

corresponding to approximately:

```text
97.4% fewer trainable parameters
```

However, this comparison does not yet include the full embedded cost of feature extraction.

---

# Why Feature Extraction Must Be Benchmarked Separately

Model size alone is not sufficient to determine the best TinyML architecture.

For example:

```text
Feature Dense INT8 classifier
Model size = 4.48 KB
```

is substantially smaller than the convolutional models.

However, its complete execution pipeline is:

```text
Raw IMU Window
      ↓
Feature Extraction
      ↓
Feature Normalization
      ↓
INT8 Quantization
      ↓
Dense Classifier
```

The computational cost of feature extraction occurs outside the neural network.

Therefore:

```text
Tiny classifier
≠
Automatically cheapest embedded system
```

The final comparison must measure:

```text
Feature extraction time
+
Preprocessing time
+
Inference time
=
Total pipeline latency
```

directly on the microcontroller.

---

# Host vs Embedded Latency

Latency measurements obtained on a desktop or notebook computer must not be interpreted as microcontroller execution time.

For example:

```text
Depthwise INT8 host inference:
~0.015 ms

Depthwise INT8 Arduino inference:
~20 ms
```

Therefore:

```text
Host latency
≠
Arduino latency
```

Host measurements are useful during development, but final architecture selection will rely on physical embedded measurements.

---

# Arduino Deployment

TensorFlow Lite Micro deployment has already been validated using the Depthwise INT8 model.

Confirmed:

```text
Model loading              PASS
Model alignment            PASS
Schema validation          PASS
Operator registration      PASS
Interpreter creation       PASS
Tensor allocation          PASS
Input tensor               PASS
Output tensor              PASS
INT8 inference             PASS
Invoke()                   PASS
Direct embedded execution  PASS
```

A model-memory alignment issue was identified during deployment.

Explicit:

```cpp
alignas(16)
```

alignment allowed the TensorFlow Lite model to execute directly from embedded program memory without requiring a duplicate model buffer in SRAM.

---

# Verified Embedded Configuration

The validated Depthwise INT8 deployment uses:

```text
Input tensor:
1 × 128 × 6 INT8

Output tensor:
1 × 4 INT8

Allocated Tensor Arena:
65,536 bytes

Observed Tensor Arena usage:
6,852 bytes

Model alignment:
16 bytes

AllocateTensors():
PASS

Invoke():
PASS

Measured Arduino inference:
~20 ms
```

This implementation provides the reference configuration for the remaining embedded benchmarks.

---

# Physical Arduino Benchmark

The next stage of Study 1 compares the optimized neural architectures under identical physical conditions.

Primary candidates:

```text
Standard CNN INT8
        vs
Depthwise CNN INT8
```

The Dense baseline and FP32 models provide reference points for the compression study.

The embedded benchmark will measure:

```text
Accuracy
Flash usage
SRAM usage
Tensor Arena usage
Model size
Preprocessing latency
Inference latency
Total latency
```

---

# Feature Engineering Benchmark

Study 2 will compare:

```text
Raw Signal
RMS Features
FFT Features
Spectral Features
```

under a controlled experimental setup.

For each representation, the following will be recorded:

```text
Input dimensionality
Classification accuracy
Parameter count
Model size
Feature extraction time
Classifier inference time
Total execution time
Flash usage
RAM usage
```

The Feature Dense INT8 model will then be evaluated as an optimized feature-based candidate.

---

# Important Interpretation of Current Accuracy

All currently evaluated models achieve:

```text
100% accuracy
```

on the current test split.

However, the test set contains only:

```text
32 windows
```

Therefore, the current results demonstrate successful classification on the existing dataset but **do not establish perfect real-world generalization**.

Further validation is required using:

```text
New IMU recordings
Live gestures
Repeated gesture executions
Different recording sessions
Potentially different users
Physical Arduino inference
```

The research objective is not simply to maximize test accuracy.

The objective is to determine which architecture provides the best:

```text
Accuracy
        +
Memory Efficiency
        +
Computational Efficiency
        +
Execution Speed
        +
Real-World Robustness
```

---

# Current Project Status

## Dataset and Preprocessing

```text
Dataset collection                         DONE
Window generation                          DONE
Fixed train/validation/test split          DONE
Train-only normalization                   DONE
```

## Study 1 — Compression Benchmark

```text
Dense Raw FP32                             DONE
Standard CNN FP32                          DONE
Standard CNN INT8                          DONE
Depthwise CNN FP32                         DONE
Depthwise CNN INT8                         DONE

Accuracy comparison                        DONE
Parameter comparison                       DONE
MAC comparison                             DONE
TFLite size comparison                     DONE
Host latency comparison                    DONE

Depthwise Arduino deployment               PASS
TensorFlow Lite Micro integration          PASS
Model alignment debugging                  DONE
Physical inference                         PASS

Standard CNN Arduino benchmark             PENDING
Final Depthwise Arduino benchmark          PENDING
Flash comparison                           PENDING
SRAM comparison                            PENDING
Total embedded latency comparison          PENDING
```

## Study 2 — Feature Engineering

```text
Feature extraction framework               DONE
72-feature combined representation         DONE
Feature Dense FP32                         DONE
Feature Dense INT8                         DONE
Feature-model host benchmarking            DONE

Controlled Raw vs RMS comparison           PENDING
Controlled Raw vs FFT comparison           PENDING
Controlled Spectral-feature comparison     PENDING

Feature Dense Arduino deployment           PENDING
Feature extraction Arduino benchmark       PENDING
Total feature-pipeline latency              PENDING
Embedded feature-memory comparison         PENDING
```

## Final Validation

```text
Live IMU comparison                        PENDING
Real-world accuracy validation             PENDING
Cross-session validation                   PENDING
Final embedded architecture selection      PENDING
```

---

# Next Milestones

The next experimental work is divided into two stages.

## Milestone 1 — Complete the Compression Benchmark

Run the optimized neural models under equivalent Arduino conditions and collect:

```text
Flash
RAM
Tensor Arena
Inference latency
Total latency
Live classification results
```

This will answer **Research Question 1**.

---

## Milestone 2 — Complete the Feature Engineering Study

Perform the controlled comparison:

```text
Raw
vs
RMS
vs
FFT
vs
Spectral Features
```

and measure both classification performance and embedded computational cost.

The Feature Dense INT8 pipeline will then be deployed and benchmarked on the Arduino.

This will answer **Research Question 2**.

---

# Final Research Goal

The final project does not aim to identify only the smallest neural network.

Instead, it aims to understand the trade-off between:

```text
Neural Network Compression
            vs
Signal Processing + Small Neural Network
```

The final analysis will determine whether resource-constrained TinyML applications benefit more from:

```text
Smaller / quantized neural architectures
```

or from:

```text
Handcrafted signal processing
+
Much smaller classifiers
```

when the **entire embedded pipeline** is considered.

---

# Final Target System

```text
Physical Gesture
      ↓
Arduino IMU
      ↓
128 × 6 Sensor Window
      ↓
Embedded Processing
      ↓
Optimized TinyML Pipeline
      ↓
INT8 Inference
      ↓
4-Class Gesture Prediction
      ↓
Confidence Score
```

Target classes:

```text
circle
left_right
rest
up_down
```

The final system will perform gesture recognition completely on-device.

---

# Project Status

**Active development — Study 1 model optimization is complete offline, Depthwise INT8 deployment is operational, and the remaining work focuses on physical Arduino benchmarking and the controlled Raw/RMS/FFT/Spectral Feature Engineering study.**
