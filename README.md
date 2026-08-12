# TinyML Compression & Feature Engineering for Gesture Recognition

Real-time IMU gesture classification and embedded TinyML optimization on the **Arduino Nano 33 BLE platform**.

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

Offline accuracy, parameter count, MAC count, and TFLite model size are available for all five compression-study models.

Physical Arduino benchmarking has now been completed for the two primary deployment candidates:

```text
Standard CNN INT8
Depthwise CNN INT8
```

These two models were tested under the same embedded configuration so that Flash, compile-reported SRAM, Tensor Arena usage, preprocessing latency, and inference latency could be compared directly.

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

Offline results:

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

INT8 quantization therefore reduced model storage without reducing accuracy on the current offline test set.

## Standard CNN INT8 — Physical Arduino Results

The final INT8 model was exported with 16-byte alignment and executed successfully using TensorFlow Lite Micro.

```text
Model size:                  10,968 bytes
Flash / program storage:     194,408 bytes
Compile-reported SRAM:       115,832 bytes
Tensor Arena allocated:      65,536 bytes
Tensor Arena used:           6,644 bytes

Average preprocessing:       ~1.39 ms
Average inference:           ~25.37 ms
Average processing time:     ~26.76 ms
```

TensorFlow Lite Micro initialization and inference both passed:

```text
Model alignment          PASS
AllocateTensors()        PASS
INT8 input/output        PASS
Invoke()                 PASS
```

### Standard CNN Live Gesture Probe

Three motion-triggered gesture classes were tested live:

```text
circle:       4 / 4 correct
left_right:   2 / 4 correct
up_down:      4 / 4 correct
```

Total for the three active gesture classes:

```text
10 / 12 correct
83.33%
```

The `rest` class was not included in this motion-triggered live test because the current runtime starts capture only after the motion score exceeds the configured threshold.

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

Offline results:

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

## Depthwise CNN INT8 — Physical Arduino Results

The final fixed-split Depthwise INT8 model was exported and benchmarked on the same Arduino setup used for the Standard CNN INT8 model.

```text
Model size:                  11,576 bytes
Flash / program storage:     205,424 bytes
Compile-reported SRAM:       115,872 bytes
Tensor Arena allocated:      65,536 bytes
Tensor Arena used:           7,156 bytes

Typical preprocessing:       ~1.45 ms
Typical inference:           ~16.88 ms
Typical processing time:     ~18.33 ms
```

TensorFlow Lite Micro execution passed:

```text
Model alignment          PASS
AllocateTensors()        PASS
INT8 input/output        PASS
Invoke()                 PASS
```

### Depthwise Live Gesture Probe

A short live robustness probe was performed using repeated physical `circle` gestures.

From the recorded trials:

```text
Ground-truth gesture:     circle
Recorded trials:          5
Correct circle outputs:   2
Misclassified outputs:    3
Observed circle accuracy: 40%
```

The incorrect predictions were primarily classified as `up_down`.

This short circle-only probe is **not directly comparable** with the Standard CNN's 12-trial three-gesture test because the live protocols were not identical. It is retained as evidence that the perfect offline test accuracy does not imply perfect real-world robustness.

# Study 1 — Compression Results

## Offline Benchmark

| Model              | Accuracy | Parameters | Approx. MACs | TFLite Size |
| ------------------ | -------: | ---------: | -----------: | ----------: |
| Dense Raw FP32     |  100.00% |     51,428 |          N/A |   203.93 KB |
| Standard CNN FP32  |  100.00% |      2,660 |      160,320 |    15.70 KB |
| Standard CNN INT8  |  100.00% |      2,660 |      160,320 |    10.71 KB |
| Depthwise CNN FP32 |  100.00% |      1,330 |       52,544 |    11.94 KB |
| Depthwise CNN INT8 |  100.00% |      1,330 |       52,544 |    11.30 KB |

All five compression-study models achieved:

```text
100% test accuracy
```

on the fixed 32-window offline test set.

## Physical Arduino Benchmark — Optimized INT8 Candidates

| Metric                  | Standard CNN INT8 | Depthwise CNN INT8 |
| ----------------------- | ----------------: | -----------------: |
| TFLite model size       |      **10.71 KB** |           11.30 KB |
| Parameters              |             2,660 |          **1,330** |
| Approx. MACs            |           160,320 |         **52,544** |
| Flash / program storage |     **194,408 B** |          205,424 B |
| Compile-reported SRAM   |     **115,832 B** |          115,872 B |
| Tensor Arena used       |       **6,644 B** |            7,156 B |
| Preprocessing latency   |          ~1.39 ms |           ~1.45 ms |
| Inference latency       |         ~25.37 ms |      **~16.88 ms** |
| Processing latency      |         ~26.76 ms |      **~18.33 ms** |

Compared with Standard CNN INT8, the Depthwise INT8 architecture provides:

```text
50.00% fewer parameters
67.23% fewer MACs
~33.5% lower Arduino inference latency
~31.5% lower preprocessing + inference latency
```

However, the final embedded binary for the Depthwise model used:

```text
~5.7% more Flash
~7.7% more Tensor Arena memory
```

and its TFLite file was slightly larger than the Standard CNN INT8 file.

Therefore, the Arduino results show an important TinyML trade-off:

```text
Lower theoretical computation
does not automatically mean
lower Flash or lower working memory.
```

## Study 1 — Current Finding

For the current Arduino deployment:

* **Depthwise CNN INT8 is the strongest compute/speed candidate** because it reduces MACs substantially and executes inference much faster.
* **Standard CNN INT8 is the stronger storage/memory candidate** among the two deployed raw-signal CNNs because it uses less Flash, a smaller Tensor Arena, and a slightly smaller TFLite file.
* Both models preserve 100% accuracy on the current offline test split.
* Short live tests reveal that offline accuracy overestimates real-world robustness.

No single optimization dominates every metric; the preferred architecture depends on whether the primary design constraint is execution speed or embedded memory footprint.

**Study 1 core compression benchmark for the two optimized deployment candidates is complete.**

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
Standard CNN INT8 host:
~0.017 ms

Standard CNN INT8 Arduino:
~25.37 ms

Depthwise INT8 host:
~0.015 ms

Depthwise INT8 Arduino:
~16.88 ms
```

Therefore:

```text
Host latency
≠
Arduino latency
```

Host measurements remain useful for development and relative profiling, but the final embedded comparison uses measurements obtained directly from the Arduino platform.

# Arduino Deployment

TensorFlow Lite Micro deployment has now been validated for both optimized INT8 convolutional candidates:

```text
Standard CNN INT8
Depthwise CNN INT8
```

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

A model-memory alignment issue identified during development was resolved using explicit:

```cpp
alignas(16)
```

alignment.

This allows the TFLite model arrays to execute directly from embedded program memory without keeping an unnecessary duplicate model buffer in SRAM.

# Verified Embedded Configuration

Both optimized INT8 models were benchmarked with the same runtime configuration:

```text
Input tensor:
1 × 128 × 6 INT8

Output tensor:
1 × 4 INT8

Allocated Tensor Arena:
65,536 bytes

Model alignment:
16 bytes

IMU accelerometer rate:
~99.84 Hz

IMU gyroscope rate:
~99.84 Hz

Motion threshold:
0.20
```

Observed Tensor Arena usage:

```text
Standard CNN INT8:
6,644 bytes

Depthwise CNN INT8:
7,156 bytes
```

Measured inference latency:

```text
Standard CNN INT8:
~25.37 ms

Depthwise CNN INT8:
~16.88 ms
```

Both models passed `AllocateTensors()` and `Invoke()` successfully on the physical Arduino.

# Physical Arduino Benchmark

The physical Arduino comparison of the two optimized raw-signal INT8 candidates is complete.

```text
Standard CNN INT8
        vs
Depthwise CNN INT8
```

The two models were evaluated under the same embedded runtime configuration using:

```text
Model size
Flash usage
Compile-reported SRAM
Tensor Arena usage
Preprocessing latency
Inference latency
Total processing latency
Live gesture behavior
```

The most important embedded result is the separation between theoretical complexity and actual embedded resource use.

Depthwise CNN INT8 reduces MAC count from:

```text
160,320
```

to:

```text
52,544
```

and reduces measured Arduino inference latency from approximately:

```text
25.37 ms
```

to:

```text
16.88 ms
```

while the Standard CNN INT8 still requires less Flash and less Tensor Arena memory.

The live gesture trials are treated as a robustness probe rather than a controlled accuracy benchmark because the two live test protocols were not identical.

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

All seven currently evaluated offline models achieve:

```text
100% accuracy
```

on the fixed test split.

However, the test set contains only:

```text
32 windows
```

and the physical Arduino tests demonstrate why this result must be interpreted carefully.

For Standard CNN INT8, the motion-triggered live probe produced:

```text
circle:       4 / 4
left_right:   2 / 4
up_down:      4 / 4

Total:
10 / 12 = 83.33%
```

For Depthwise CNN INT8, the recorded circle-only probe produced:

```text
circle:
2 / 5 correct = 40%
```

The two percentages should **not** be used as a direct head-to-head live accuracy comparison because the test protocols and class coverage were different.

Instead, they demonstrate a broader result:

```text
Perfect offline test accuracy
does not imply
perfect real-world gesture recognition.
```

Further robustness evaluation could include:

```text
New IMU recordings
Repeated gesture executions
Different recording sessions
Different users
Balanced live trials
```

These improvements are not required to continue the current optimization study, but the limitation is documented explicitly.

The main research objective remains the comparison of:

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

Standard CNN INT8 Arduino export           DONE
Standard CNN INT8 Arduino deployment       PASS
Standard CNN INT8 Flash measurement        DONE
Standard CNN INT8 SRAM measurement         DONE
Standard CNN INT8 Tensor Arena             DONE
Standard CNN INT8 inference timing         DONE
Standard CNN live gesture probe            DONE

Depthwise CNN INT8 Arduino export          DONE
Depthwise CNN INT8 Arduino deployment      PASS
Depthwise INT8 Flash measurement           DONE
Depthwise INT8 SRAM measurement            DONE
Depthwise INT8 Tensor Arena                DONE
Depthwise INT8 inference timing            DONE
Depthwise live gesture probe               DONE

Embedded INT8 candidate comparison         DONE
Study 1 core benchmark                     DONE
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
Total feature-pipeline latency             PENDING
Embedded feature-memory comparison         PENDING
```

## Final Analysis

```text
Study 1 compression result                 DONE
Study 2 feature-engineering result          PENDING
Cross-study comparison                     PENDING
Final embedded architecture selection      PENDING
README final cleanup                       PENDING
Final thesis figures / presentation        PENDING
```

# Next Milestones

## Milestone 1 — Compression Benchmark

**COMPLETE ✅**

The optimized Standard CNN INT8 and Depthwise CNN INT8 models have now been exported, deployed, and benchmarked on the physical Arduino under the same runtime configuration.

The main Study 1 finding is:

```text
Standard CNN INT8
→ lower Flash / Tensor Arena / model size

Depthwise CNN INT8
→ fewer parameters / fewer MACs / faster inference
```

The next work therefore moves to Study 2.

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

and measure both classification performance and computational cost.

For each representation, record:

```text
Input dimensionality
Accuracy
Parameter count
Model size
Feature extraction time
Classifier inference time
Total pipeline time
```

The Feature Dense INT8 pipeline will then be deployed and benchmarked on the Arduino so that the total cost of:

```text
Signal processing
+
Feature normalization
+
INT8 classifier
```

can be compared with the raw-signal CNN pipelines.

This will answer **Research Question 2**.

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

**Active development — Study 1 core compression benchmarking is complete, including physical Arduino measurements for Standard CNN INT8 and Depthwise CNN INT8. The project is now moving to Study 2: the controlled Raw/RMS/FFT/Spectral Feature Engineering comparison and the final Feature Dense INT8 embedded benchmark.**
