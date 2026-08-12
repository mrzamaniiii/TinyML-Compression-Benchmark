# TinyML Gesture Classification & Model Optimization on Arduino Nano 33 BLE

Real-time IMU gesture classification and TinyML model optimization for resource-constrained embedded deployment.

This project develops and compares multiple neural-network architectures for classifying hand gestures from six-axis IMU data and evaluates their suitability for deployment on an Arduino Nano 33 BLE using TensorFlow Lite for Microcontrollers.

The project covers:

* IMU data acquisition
* Gesture dataset generation
* Train/validation/test splitting
* Train-only normalization
* Dense neural-network baseline
* Standard 1D CNN
* Full INT8 quantization
* Depthwise Separable CNN
* Feature engineering
* Feature-based neural classification
* TensorFlow Lite conversion
* Model-size benchmarking
* Parameter-count comparison
* MAC-count estimation
* Host inference-latency benchmarking
* Arduino deployment
* Embedded memory debugging
* On-device inference
* TinyML architecture comparison

---

# Project Overview

The objective is not only to obtain high gesture-classification accuracy, but to determine which model architecture provides the best trade-off between:

```text
Accuracy
Model size
Parameter count
Computational complexity
Inference latency
Embedded memory usage
```

The system uses six-axis IMU measurements:

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

or:

```text
768 raw sensor values
```

Four gesture classes are considered:

```text
circle
left_right
rest
up_down
```

---

# Dataset

The current dataset split is:

```text
Training samples:     256
Validation samples:    32
Test samples:          32
```

Input shape:

```text
128 × 6
```

The test set contains eight samples from each gesture class.

Normalization statistics are calculated using the **training set only** to avoid information leakage from the validation or test sets.

---

# Gesture Classes

```text
0 → circle
1 → left_right
2 → rest
3 → up_down
```

The same class ordering is preserved during training, evaluation, TFLite conversion, and embedded deployment.

---

# Model Optimization Study

Seven model configurations have currently been evaluated.

## Model 1 — Dense Raw FP32

A fully connected baseline operating directly on the flattened raw IMU window.

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

This model provides a useful baseline but is highly inefficient for embedded deployment compared with the later architectures.

---

# Model 2 — Standard 1D CNN FP32

A temporal convolutional network operating directly on the raw IMU window.

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

Compared with the Dense baseline, the CNN dramatically reduces parameter count and model size while preserving test accuracy.

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

Model-size reduction compared with the FP32 Standard CNN:

```text
31.78%
```

Quantization therefore reduced storage requirements without reducing test accuracy.

---

# Model 4 — Depthwise Separable CNN FP32

A more computationally efficient convolutional architecture was then evaluated using separable convolutions.

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
Parameter reduction:   50.00%
MAC reduction:         67.23%
Model-size reduction:  23.96%
```

while maintaining the same test accuracy.

---

# Model 5 — Depthwise CNN Full INT8

The Depthwise Separable CNN was subsequently converted to Full INT8.

Results:

```text
Float accuracy:      100.00%
INT8 accuracy:       100.00%
Accuracy drop:       0.00 percentage points
Parameters:          1,330
Approx. MACs:        52,544
TFLite INT8 size:    11.30 KB
Average confidence:  0.996094
Host mean latency:   0.0150 ms
```

INT8 size reduction compared with the FP32 Depthwise model:

```text
5.32%
```

The model was also successfully deployed and executed using TensorFlow Lite Micro on the physical Arduino platform.

A previously measured on-device inference latency was approximately:

```text
~20.2 ms
```

This value represents physical Arduino execution and should not be directly compared with host-PC latency measurements.

---

# Model 6 — Feature Engineering + Dense FP32

The next optimization stage investigated whether handcrafted statistical features could replace most of the neural-network computation.

Each raw window:

```text
128 × 6
```

is transformed into:

```text
72 engineered features
```

before classification.

The neural classifier is:

```text
72
 ↓
Dense 16
 ↓
Dense 8
 ↓
Dense 4
```

Results:

```text
Test accuracy:          100.00%
Parameters:             1,340
Classifier MACs:        1,312
TFLite FP32 size:       7.45 KB
Average confidence:     0.999535
Host feature latency:   0.98795 ms
Host classifier latency:0.00229 ms
Host total pipeline:    0.99024 ms
```

Important:

```text
Classifier MAC count does NOT include feature-extraction computation.
```

Therefore, comparisons involving the 1,312 MAC figure must distinguish neural-classifier computation from complete pipeline computation.

---

# Model 7 — Feature Engineering + Dense Full INT8

The feature-based Dense classifier was then converted to Full INT8.

Input:

```text
1 × 72
```

Output:

```text
1 × 4
```

Results:

```text
Float accuracy:         100.00%
INT8 accuracy:          100.00%
Accuracy drop:          0.00 percentage points

Parameters:             1,340
Classifier MACs:        1,312
TFLite INT8 size:       4.48 KB
Average confidence:     0.996094

Host feature latency:   1.04228 ms
Host classifier latency:0.00261 ms
Host total pipeline:    1.04489 ms
```

Model-size reduction compared with Feature Dense FP32:

```text
39.81%
```

The confusion matrix remained perfect after quantization:

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

for all four classes on the current 32-sample test set.

---

# Current TinyML Benchmark

| Model                  |    Accuracy | Parameters | Approx. MACs | TFLite Size |
| ---------------------- | ----------: | ---------: | -----------: | ----------: |
| Dense Raw FP32         |     100.00% |     51,428 |          N/A |   203.93 KB |
| Standard CNN FP32      |     100.00% |      2,660 |      160,320 |    15.70 KB |
| Standard CNN INT8      |     100.00% |      2,660 |      160,320 |    10.71 KB |
| Depthwise CNN FP32     |     100.00% |      1,330 |       52,544 |    11.94 KB |
| Depthwise CNN INT8     |     100.00% |      1,330 |       52,544 |    11.30 KB |
| Feature Dense FP32     |     100.00% |      1,340 |       1,312* |     7.45 KB |
| **Feature Dense INT8** | **100.00%** |  **1,340** |   **1,312*** | **4.48 KB** |

* Feature-based MAC counts refer only to the neural classifier and do not include feature-extraction computation.

---

# Compression Results

Relative to the original Dense Raw FP32 baseline:

```text
Dense Raw FP32:
203.93 KB

Feature Dense INT8:
4.48 KB
```

This corresponds to approximately:

```text
97.8% reduction in TFLite model size
```

while maintaining:

```text
100% accuracy
```

on the current test set.

Parameter count was reduced from:

```text
51,428
```

to:

```text
1,340
```

which corresponds to approximately:

```text
97.4% fewer trainable parameters
```

for the feature-based model.

---

# Important Interpretation of the Results

All seven evaluated models currently achieve:

```text
100% test accuracy
```

However, the test set contains only:

```text
32 samples
```

Therefore, these results demonstrate successful classification on the current dataset but should not yet be interpreted as proof of perfect real-world generalization.

Additional evaluation is required using:

```text
Live IMU data
New recording sessions
Different gesture executions
Repeated trials
Potentially different users
On-device inference
```

This distinction is important because the primary objective of the optimization study is not simply to maximize accuracy on a small test set.

The objective is to identify the architecture that maintains classification performance while minimizing embedded resource requirements.

---

# Current Best Candidate

Based on the current offline benchmark, the strongest resource-efficiency candidate is:

```text
Feature Engineering
       ↓
72 features
       ↓
INT8 Dense classifier
       ↓
4 gesture classes
```

with:

```text
Accuracy:          100.00%
Parameters:        1,340
Classifier MACs:   1,312
TFLite size:       4.48 KB
```

However, the final embedded-model selection will depend on physical Arduino measurements.

In particular, feature extraction introduces computation outside the neural network.

Therefore the final comparison must measure the **complete pipeline**, not only the neural-network inference call.

---

# Arduino Deployment Benchmark

The next experimental stage compares the most relevant optimized models directly on the physical Arduino Nano 33 BLE.

The primary deployment candidates are:

```text
Model 3
Standard CNN INT8

Model 5
Depthwise CNN INT8

Model 7
Feature Dense INT8
```

For each architecture, the following quantities will be measured:

```text
Test / live accuracy
Flash usage
SRAM usage
Tensor Arena usage
Model size
Preprocessing latency
Feature-extraction latency
Neural-network inference latency
Total pipeline latency
```

This will allow the final model to be selected based on actual embedded performance rather than host-PC measurements alone.

---

# Host vs Arduino Latency

Host latency measurements reported during model development must not be interpreted as microcontroller latency.

For example:

```text
Depthwise INT8 host:
~0.015 ms

Previously measured Arduino inference:
~20.2 ms
```

These measurements were obtained on fundamentally different hardware.

Therefore:

```text
Host latency
≠
Arduino latency
```

The final performance comparison will rely on measurements obtained directly from the physical embedded platform.

---

# Embedded Deployment Status

The TensorFlow Lite Micro deployment pipeline has already been validated using the Depthwise INT8 model.

Confirmed on the physical microcontroller:

```text
Model loading             PASS
Model alignment           PASS
Schema validation         PASS
Operator registration     PASS
Interpreter creation      PASS
Tensor allocation         PASS
Input tensor              PASS
Output tensor             PASS
INT8 inference            PASS
Invoke()                  PASS
Direct model execution    PASS
```

A critical deployment issue involving embedded model-memory alignment was identified and resolved.

Explicit:

```cpp
alignas(16)
```

alignment allowed the TFLite model to execute directly from embedded memory without maintaining a duplicate model in SRAM.

---

# Verified Depthwise Embedded Configuration

The previously validated Depthwise INT8 configuration used:

```text
Input:
1 × 128 × 6 INT8

Output:
1 × 4 INT8

Tensor Arena:
65,536 bytes allocated

Observed Tensor Arena usage:
6,852 bytes

Model alignment:
16-byte aligned

AllocateTensors():
PASS

Invoke():
PASS

Measured Arduino inference:
~20 ms
```

This deployment establishes a working reference implementation for the upcoming multi-model Arduino benchmark.

---

# Current Project Status

```text
Dataset collection                     DONE
Train/validation/test split            DONE
Train-only normalization               DONE

Model 1 Dense Raw FP32                 DONE
Model 2 Standard CNN FP32              DONE
Model 3 Standard CNN INT8              DONE
Model 4 Depthwise CNN FP32             DONE
Model 5 Depthwise CNN INT8             DONE
Model 6 Feature Dense FP32             DONE
Model 7 Feature Dense INT8             DONE

Model accuracy comparison              DONE
Parameter comparison                   DONE
MAC comparison                         DONE
TFLite size comparison                 DONE
Host latency measurements              DONE

Depthwise Arduino deployment           PASS
TensorFlow Lite Micro integration      PASS
Model alignment debugging              DONE
Physical inference                     PASS

Standard CNN INT8 Arduino benchmark    PENDING
Depthwise INT8 final benchmark         PENDING
Feature Dense INT8 Arduino benchmark   PENDING

Flash comparison                       PENDING
SRAM comparison                        PENDING
Total pipeline latency comparison      PENDING
Live IMU comparison                    PENDING
Real-world accuracy validation         PENDING

Final embedded model selection         PENDING
```

---

# Next Milestone

The next milestone is:

## Physical Arduino Model Benchmark

The three optimized INT8 architectures will be evaluated under the same embedded conditions:

```text
Standard CNN INT8
        vs
Depthwise CNN INT8
        vs
Feature Dense INT8
```

The goal is to determine whether the extremely small Feature Dense INT8 classifier remains the most efficient architecture after including the computational cost of feature extraction.

The final decision will therefore be based on:

```text
Accuracy
+
Flash
+
SRAM
+
Model size
+
Total latency
+
Real-world robustness
```

rather than model size alone.

---

# Final Target

The final embedded pipeline will operate completely on-device:

```text
Physical gesture
      ↓
Arduino IMU
      ↓
128 × 6 sensor window
      ↓
Embedded preprocessing
      ↓
Optimized TinyML pipeline
      ↓
INT8 inference
      ↓
4-class prediction
      ↓
Confidence score
```

Target classes:

```text
circle
left_right
rest
up_down
```

No external computer or cloud inference service will be required after deployment.

---

# About

TinyML research and development project investigating neural-network compression and architecture optimization for real-time IMU gesture recognition on the Arduino Nano 33 BLE.

The project compares Dense networks, standard 1D CNNs, Depthwise Separable CNNs, INT8 quantization, and feature-engineered classifiers to determine the most resource-efficient architecture for physical microcontroller deployment.

Current results demonstrate that model size can be reduced from approximately:

```text
203.93 KB
```

to:

```text
4.48 KB
```

while preserving 100% accuracy on the current test set.

The next stage evaluates whether these offline efficiency gains translate into lower Flash usage, SRAM consumption, and total execution latency on the physical Arduino platform.

---

# Project Status

**Active development — model optimization benchmark complete; physical multi-model Arduino benchmarking is the next stage.**
