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

Physical Arduino benchmarking was completed for the two optimized raw-signal INT8 candidates:

```text
Standard CNN INT8
Depthwise CNN INT8
```

---

## Study 2 — TinyML Feature Engineering Study

### Research Question

> Is handcrafted signal feature extraction still useful for TinyML when compared with learning directly from raw sensor signals?

The same gesture-classification problem was evaluated using four signal representations:

```text
Raw IMU Signal
      vs
RMS Features
      vs
FFT Features
      vs
Spectral Features
```

The controlled comparison used the same samples, the same fixed train/validation/test split, train-only normalization, and the same classifier architecture for all four representations.

The strongest feature representation was then used to create an optimized embedded classifier.

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
Signal Processing / Preprocessing
      ↓
TinyML Classifier
      ↓
INT8 Inference
      ↓
Gesture Prediction
      ↓
Confidence Score
```

The final system operates completely on-device without cloud inference or an external computer.

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

Each raw input window contains:

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

The same class ordering is preserved during dataset generation, training, evaluation, TFLite conversion, and embedded deployment.

---

# Dataset

The dataset contains:

```text
320 total windows
80 windows per gesture class
```

The final controlled experiments use a fixed:

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

All normalization statistics are calculated using the **training set only** to prevent validation or test information from leaking into the training pipeline.

---

# Study 1 — TinyML Compression Benchmark

The five core compression-study architectures are:

```text
Model 1 — Dense Raw FP32
Model 2 — Standard CNN FP32
Model 3 — Standard CNN INT8
Model 4 — Depthwise CNN FP32
Model 5 — Depthwise CNN INT8
```

Two additional feature-based models were developed during exploration:

```text
Model 6 — Feature Dense FP32
Model 7 — Feature Dense INT8
```

---

# Model 1 — Dense Raw FP32

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

The Dense model provides a high-accuracy baseline but is inefficient for embedded deployment.

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

---

# Model 3 — Standard CNN Full INT8

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

## Standard CNN INT8 — Physical Arduino Results

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

TensorFlow Lite Micro:

```text
Model alignment          PASS
AllocateTensors()        PASS
INT8 input/output        PASS
Invoke()                 PASS
```

### Standard CNN Live Gesture Probe

```text
circle:       4 / 4 correct
left_right:   2 / 4 correct
up_down:      4 / 4 correct

Total:
10 / 12 = 83.33%
```

The `rest` class was not included in this motion-triggered live probe because capture begins only after the motion score exceeds the threshold.

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

TensorFlow Lite Micro:

```text
Model alignment          PASS
AllocateTensors()        PASS
INT8 input/output        PASS
Invoke()                 PASS
```

### Depthwise Live Gesture Probe

A short circle-only robustness probe produced:

```text
Ground-truth gesture:     circle
Recorded trials:          5
Correct circle outputs:   2
Misclassified outputs:    3
Observed circle accuracy: 40%
```

The incorrect predictions were primarily classified as `up_down`.

This result is **not directly comparable** with the Standard CNN live percentage because the two live protocols were not identical.

---

# Study 1 — Compression Results

## Offline Benchmark

| Model | Accuracy | Parameters | Approx. MACs | TFLite Size |
|---|---:|---:|---:|---:|
| Dense Raw FP32 | 100.00% | 51,428 | N/A | 203.93 KB |
| Standard CNN FP32 | 100.00% | 2,660 | 160,320 | 15.70 KB |
| Standard CNN INT8 | 100.00% | 2,660 | 160,320 | 10.71 KB |
| Depthwise CNN FP32 | 100.00% | 1,330 | 52,544 | 11.94 KB |
| Depthwise CNN INT8 | 100.00% | 1,330 | 52,544 | 11.30 KB |

All five compression-study models achieved 100% accuracy on the fixed 32-window offline test set.

## Physical Arduino Benchmark — Optimized INT8 Candidates

| Metric | Standard CNN INT8 | Depthwise CNN INT8 |
|---|---:|---:|
| TFLite model size | **10.71 KB** | 11.30 KB |
| Parameters | 2,660 | **1,330** |
| Approx. MACs | 160,320 | **52,544** |
| Flash / program storage | **194,408 B** | 205,424 B |
| Compile-reported SRAM | **115,832 B** | 115,872 B |
| Tensor Arena used | **6,644 B** | 7,156 B |
| Preprocessing latency | ~1.39 ms | ~1.45 ms |
| Inference latency | ~25.37 ms | **~16.88 ms** |
| Processing latency | ~26.76 ms | **~18.33 ms** |

Compared with Standard CNN INT8, Depthwise CNN INT8 provides:

```text
50.00% fewer parameters
67.23% fewer MACs
~33.5% lower Arduino inference latency
~31.5% lower preprocessing + inference latency
```

However, it uses slightly more Flash, Tensor Arena memory, and TFLite storage.

### Study 1 Finding

- **Depthwise CNN INT8** is the stronger compute/speed candidate.
- **Standard CNN INT8** is the stronger storage/memory candidate among the two deployed raw-signal CNNs.
- Lower theoretical computation does not automatically imply lower Flash or lower working memory.
- Perfect offline accuracy does not imply perfect real-world robustness.

**Study 1 — COMPLETE ✅**

---

# Study 2 — TinyML Feature Engineering Study

Study 2 investigates whether classical signal processing can reduce the amount of information presented to the neural network while preserving classification performance.

The controlled comparison uses:

```text
Raw Signal
RMS Features
FFT Features
Spectral Features
```

All four representations use:

```text
The same 320 windows
The same 80/10/10 split
The same train/validation/test indices
Train-only normalization
The same Dense classifier architecture
```

This isolates the effect of the input representation.

---

# Study 2.1 — Controlled Feature Representation Comparison

The representations are:

```text
Raw:
128 × 6 → Flatten → 768 values

RMS:
1 RMS value per IMU channel → 6 features

FFT:
8 non-DC magnitude bins per channel → 48 features

Spectral:
Dominant frequency + peak PSD per channel → 12 features
```

Results:

| Representation | Input Features | Parameters | Test Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Raw | 768 | 51,428 | **100.00%** | **1.0000** |
| RMS | **6** | **2,660** | **100.00%** | **1.0000** |
| FFT | 48 | 5,348 | **100.00%** | **1.0000** |
| Spectral | 12 | 3,044 | 96.875% | 0.9686 |

The Spectral representation misclassified one of the 32 test samples.

Its confusion matrix was:

```text
[[8 0 0 0]
 [1 7 0 0]
 [0 0 8 0]
 [0 0 0 8]]
```

The most important result is that RMS reduced the input dimensionality from:

```text
768 raw values
```

to:

```text
6 RMS features
```

while preserving 100% accuracy on the current test split.

**Study 2.1 — COMPLETE ✅**

---

# Study 2.2 — Host Efficiency Benchmark

FP32 TFLite model size, feature extraction time, classifier inference time, and total host pipeline time were measured.

| Representation | Features | Parameters | Accuracy | FP32 TFLite Size | Feature Time | Classifier Time | Total Pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 768 | 51,428 | 100.00% | 202.94 KB | 0.00076 ms | 0.01041 ms | **0.01116 ms** |
| RMS | **6** | **2,660** | **100.00%** | **12.44 KB** | 0.18315 ms | 0.00619 ms | 0.18935 ms |
| FFT | 48 | 5,348 | 100.00% | 22.94 KB | 0.19608 ms | 0.00397 ms | 0.20005 ms |
| Spectral | 12 | 3,044 | 96.875% | 13.94 KB | 3.40530 ms | 0.00411 ms | 3.40942 ms |

Interpretation:

- Raw is fastest on the host because it performs almost no feature extraction, but it requires by far the largest classifier.
- RMS provides the strongest accuracy/size trade-off.
- FFT preserves accuracy but requires more features and a larger classifier than RMS.
- Spectral features are the most expensive to extract and also reduce accuracy on the current test split.

Host latency is not used as a substitute for Arduino latency.

**Study 2.2 — COMPLETE ✅**

---

# Study 2.3 — RMS Tiny Classifier

Because RMS was the strongest feature representation, a much smaller classifier was created:

```text
6 RMS Features
      ↓
Dense 16
      ↓
Dense 8
      ↓
Dense 4
```

This network contains only:

```text
284 trainable parameters
```

Results:

```text
Keras FP32 accuracy:       100.00%
Full INT8 accuracy:        100.00%
Accuracy drop:             0.00 percentage points

FP32 TFLite size:          3,236 bytes
FP32 TFLite size:          3.160 KB

INT8 TFLite size:          3,432 bytes
INT8 TFLite size:          3.352 KB

Host INT8 classifier:      0.003758 ms
```

INT8 confusion matrix:

```text
[[8 0 0 0]
 [0 8 0 0]
 [0 0 8 0]
 [0 0 0 8]]
```

An important result is that the INT8 file is slightly larger than the FP32 file:

```text
FP32: 3.160 KB
INT8: 3.352 KB
```

For this extremely small network, quantization metadata overhead exceeds the storage saved by using INT8 weights.

Therefore:

```text
Quantization
does not necessarily reduce
TFLite file size
for extremely small networks.
```

**Study 2.3 — COMPLETE ✅**

---

# Study 2.4 — RMS Tiny INT8 Arduino Deployment

The RMS Tiny INT8 model was exported using 16-byte model alignment and deployed successfully using TensorFlow Lite Micro.

Embedded pipeline:

```text
128 × 6 IMU Window
      ↓
RMS Extraction
      ↓
6 RMS Features
      ↓
Feature Standardization
      ↓
INT8 Quantization
      ↓
284-Parameter Dense Classifier
      ↓
Prediction
```

TensorFlow Lite Micro validation:

```text
Model alignment          PASS
AllocateTensors()        PASS
INT8 input/output        PASS
Invoke()                 PASS
```

Quantization parameters:

```text
Input:
1 × 6 INT8

Input scale:
0.01591838

Input zero point:
-33

Output:
1 × 4 INT8

Output scale:
0.00390625

Output zero point:
-128
```

Physical Arduino resource measurements:

```text
TFLite model size:           3,432 bytes
TFLite model size:           3.352 KB

Flash / program storage:     165,960 bytes
Compile-reported SRAM:       115,640 bytes

Tensor Arena allocated:      65,536 bytes
Tensor Arena used:           948 bytes
```

The compile-reported SRAM values of the three Arduino benchmarks are similar because the benchmark sketches intentionally reserve the same 64 KB Tensor Arena. For working-memory comparison, `arena_used_bytes()` is therefore the more informative metric.

**Study 2.4 — COMPLETE ✅**

---

# Study 2.5 — RMS Embedded Latency Benchmark

Three physical circle trials were recorded.

| Trial | RMS Extraction | Feature Preprocessing | INT8 Inference | Total Processing | Prediction |
|---|---:|---:|---:|---:|---|
| 1 | 181 µs | 20 µs | 165 µs | 366 µs | circle |
| 2 | 165 µs | 20 µs | 165 µs | 350 µs | circle |
| 3 | 137 µs | 20 µs | 165 µs | 322 µs | circle |

Average embedded processing:

```text
RMS extraction:             ~161 µs
Feature preprocessing:       20 µs
INT8 inference:             165 µs
Total processing:           ~346 µs
Total processing:           ~0.346 ms
```

Live circle probe:

```text
3 / 3 correct

Confidence:
0.996094 for all three recorded trials
```

The live probe is intentionally treated as a limited robustness check rather than a controlled accuracy benchmark.

**Study 2.5 — COMPLETE ✅**

---

# Study 2 — Final Finding

Study 2 provides evidence that handcrafted feature extraction can still be highly valuable in TinyML when the extracted representation matches the structure of the task.

For the current gesture-classification dataset:

```text
Raw Input:
768 values
51,428-parameter baseline classifier

RMS Input:
6 features
284-parameter optimized classifier
```

The RMS Tiny INT8 model preserved:

```text
100% offline test accuracy
```

while requiring:

```text
3.35 KB TFLite model
948 bytes Tensor Arena
165,960 bytes total sketch Flash
~0.165 ms classifier inference
~0.346 ms complete RMS processing pipeline
```

This demonstrates that classical signal processing can move useful structure out of the neural network and allow the classifier itself to become substantially smaller.

However, this conclusion applies to the current dataset and experimental setup. The test set contains only 32 windows, and the live trials are limited. Broader generalization would require additional sessions, users, and balanced live evaluation.

**Study 2 — COMPLETE ✅**

---

# Cross-Study Embedded Comparison

The three most relevant embedded INT8 candidates are:

```text
Standard CNN INT8
Depthwise CNN INT8
RMS Tiny INT8
```

| Metric | Standard CNN INT8 | Depthwise CNN INT8 | RMS Tiny INT8 |
|---|---:|---:|---:|
| Offline Accuracy | 100.00% | 100.00% | **100.00%** |
| Input Representation | Raw 128×6 | Raw 128×6 | **6 RMS features** |
| Parameters | 2,660 | 1,330 | **284** |
| TFLite Model Size | 10.71 KB | 11.30 KB | **3.35 KB** |
| Flash / Program Storage | 194,408 B | 205,424 B | **165,960 B** |
| Compile-Reported SRAM* | 115,832 B | 115,872 B | **115,640 B** |
| Tensor Arena Used | 6,644 B | 7,156 B | **948 B** |
| Feature / Preprocess Time | ~1.39 ms | ~1.45 ms | **~0.181 ms** |
| Inference Time | ~25.37 ms | ~16.88 ms | **0.165 ms** |
| Total Processing Time | ~26.76 ms | ~18.33 ms | **~0.346 ms** |

\* All three benchmark sketches reserve a 64 KB Tensor Arena, so compile-reported SRAM should not be interpreted as the actual working-memory requirement of the model itself.

The RMS Tiny INT8 pipeline provides the strongest overall resource-efficiency result in the current project.

Compared with Standard CNN INT8, it uses:

```text
~68.7% smaller TFLite model
~14.6% less total sketch Flash
~85.7% less Tensor Arena
~99.3% lower neural inference latency
~98.7% lower total processing latency
```

Compared with Depthwise CNN INT8, it uses:

```text
~70.3% smaller TFLite model
~19.2% less total sketch Flash
~86.8% less Tensor Arena
~99.0% lower neural inference latency
~98.1% lower total processing latency
```

---

# Final Research Interpretation

The two studies answer different but complementary questions.

## Study 1

Neural-network architecture optimization provides meaningful gains.

```text
Standard CNN INT8
→ smaller Flash / Arena footprint among the raw CNN candidates

Depthwise CNN INT8
→ fewer MACs and faster raw-signal inference
```

No single neural-network compression technique dominates every metric.

## Study 2

Signal-processing-based feature engineering can outperform neural-network-only compression when a compact representation preserves task-relevant information.

For the current dataset:

```text
RMS Feature Engineering
+
Tiny INT8 Dense Classifier
```

produces the strongest overall embedded resource-efficiency result.

The project therefore demonstrates an important TinyML principle:

```text
The smallest neural network
is not always obtained only by
compressing the neural network.

Sometimes the larger gain comes from
changing the representation
before the network.
```

---

# Important Interpretation of Accuracy

All final offline deployment candidates achieve:

```text
100% accuracy
```

on the current fixed 32-window test set.

This does **not** establish perfect real-world generalization.

The live tests show that behavior outside the offline test split can differ.

Further robustness evaluation could include:

```text
New IMU recordings
Different recording sessions
Different users
Balanced live trials
Cross-session validation
```

The current work should therefore be interpreted primarily as a controlled TinyML architecture and embedded-efficiency study.

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

Standard CNN INT8 Arduino deployment       DONE
Standard CNN embedded benchmark            DONE

Depthwise CNN INT8 Arduino deployment      DONE
Depthwise embedded benchmark               DONE

Study 1 result                             DONE
```

## Study 2 — Feature Engineering

```text
Raw vs RMS comparison                      DONE
Raw vs FFT comparison                      DONE
Spectral-feature comparison                DONE
Controlled feature benchmark               DONE
Host efficiency benchmark                  DONE

RMS Tiny classifier                        DONE
RMS Tiny Full INT8                         DONE
RMS Tiny Arduino export                    DONE
RMS Tiny Arduino deployment                DONE
RMS embedded memory benchmark              DONE
RMS embedded latency benchmark             DONE

Study 2 result                             DONE
```

## Final Analysis

```text
Study 1 compression result                 DONE
Study 2 feature-engineering result         DONE
Cross-study embedded comparison            DONE
Final embedded architecture selection      DONE
README technical update                    DONE

GitHub RMS Arduino files                   PENDING
Final figures / plots                      PENDING
Final presentation / thesis integration    PENDING
```

---

# Final Embedded Architecture Selection

Based on the current controlled experiments and physical Arduino measurements, the strongest overall resource-efficiency candidate is:

```text
128 × 6 IMU Window
      ↓
6 RMS Features
      ↓
Tiny Dense Classifier
6 → 16 → 8 → 4
      ↓
Full INT8 Inference
      ↓
Gesture Prediction
```

Key embedded results:

```text
Parameters:             284
TFLite model size:      3.35 KB
Flash:                  165,960 bytes
Tensor Arena used:      948 bytes
Inference:              ~0.165 ms
Total processing:       ~0.346 ms
Offline test accuracy:  100%
```

This architecture is selected as the final embedded candidate for the current project.

---

# Repository Status

The Arduino directory should contain the final deployed implementations:

```text
Arduino/
├── Standard_CNN_INT8_Gesture_Classifier.ino
├── standard_cnn_model.h
├── standard_cnn_scaler.h
│
├── Depthwise_CNN_INT8_Gesture_Classifier.ino
├── depthwise_cnn_model.h
├── depthwise_cnn_scaler.h
│
├── RMS_Tiny_INT8_Gesture_Classifier.ino
├── rms_tiny_model.h
└── rms_feature_scaler.h
```

---

# Final Target System

```text
Physical Gesture
      ↓
Arduino IMU
      ↓
128 × 6 Sensor Window
      ↓
RMS Feature Extraction
      ↓
6 RMS Features
      ↓
INT8 Tiny Classifier
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

The final system performs gesture recognition completely on-device.

---

# Project Status

**Active development — Study 1 and Study 2 are complete. The final embedded architecture has been selected based on physical Arduino measurements. Remaining work is focused on repository cleanup, final figures, and presentation/thesis integration.**
