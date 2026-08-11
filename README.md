# TinyML Gesture Classification on Arduino Nano 33 BLE

Real-time IMU gesture classification using an INT8 Depthwise CNN deployed with TensorFlow Lite for Microcontrollers.

This project implements an end-to-end TinyML pipeline for classifying hand gestures from six-axis IMU data directly on an Arduino-class microcontroller.

The project covers:

- IMU data acquisition
- Dataset generation
- Gesture classification
- Depthwise CNN training
- INT8 quantization
- Embedded model export
- Raw sensor standardization
- TensorFlow Lite Micro deployment
- Memory and alignment debugging
- On-device inference
- Real-time gesture classification integration

---

## Table of Contents

- [Project Overview](#project-overview)
- [Gesture Classes](#gesture-classes)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Input](#model-input)
- [Depthwise CNN](#depthwise-cnn)
- [INT8 Quantization](#int8-quantization)
- [Raw IMU Scaling](#raw-imu-scaling)
- [Embedded Deployment](#embedded-deployment)
- [TensorFlow Lite Micro Operators](#tensorflow-lite-micro-operators)
- [Tensor Arena](#tensor-arena)
- [Deployment Debugging](#deployment-debugging)
- [Controlled Test 3](#controlled-test-3--minimal-tflm-runtime)
- [Controlled Test 4](#controlled-test-4--raw-scaler-reintroduced)
- [Controlled Test 5](#controlled-test-5--aligned-ram-model)
- [Controlled Test 6](#controlled-test-6--direct-aligned-flash-model)
- [Root Cause](#root-cause)
- [Memory Usage](#memory-usage)
- [Inference Performance](#inference-performance)
- [Confirmed Results](#confirmed-results)
- [Current Project Status](#current-project-status)
- [Next Development Stage](#next-development-stage)
- [Repository Structure](#repository-structure)
- [Development Strategy](#development-strategy)
- [About](#about)

---

# Project Overview

The goal of this project is to build a lightweight gesture-classification system capable of running directly on a microcontroller using TinyML.

Six-axis IMU measurements are used as the input:

```text
ax
ay
az
gx
gy
gz
```

where:

```text
ax, ay, az = accelerometer measurements
gx, gy, gz = gyroscope measurements
```

The neural network processes a temporal window containing:

```text
128 samples × 6 channels
```

which gives:

```text
768 input values
```

for every inference.

The final intended embedded pipeline is:

```text
On-board IMU
      ↓
ax, ay, az, gx, gy, gz
      ↓
128 × 6 sample window
      ↓
Raw sensor standardization
      ↓
INT8 quantization
      ↓
Depthwise CNN
      ↓
TensorFlow Lite Micro
      ↓
4-class output
      ↓
Gesture prediction
      ↓
Confidence score
```

A major goal of the project is to perform the complete inference pipeline locally on the microcontroller without requiring a computer or cloud-based inference service.

---

# Gesture Classes

The project contains four gesture categories:

```text
circle
left_right
rest
up_down
```

The neural network therefore produces four output values, one for each gesture class.

The final embedded classifier will map the model output to:

```text
0 → circle
1 → left_right
2 → rest
3 → up_down
```

The class ordering used by the final firmware must remain identical to the class ordering used during model training.

---

# System Architecture

The final system architecture is:

```text
┌──────────────────────┐
│    On-board IMU      │
│ Accelerometer + Gyro │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 6-axis sensor stream │
│ ax ay az gx gy gz    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 128-sample window    │
│ 128 × 6 = 768 values │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Standardization      │
│ raw_mean / raw_std   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ INT8 Quantization    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Depthwise CNN        │
│ TensorFlow Lite      │
│ Micro                │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4-Class Prediction   │
└──────────┬───────────┘
           │
           ▼
 circle / left_right
   rest / up_down
```

---

# Dataset

The gesture dataset was collected using the Arduino's on-board IMU.

Each recorded sample contains six sensor channels:

```text
ax
ay
az
gx
gy
gz
```

The dataset therefore contains both linear acceleration and angular velocity information.

The Depthwise CNN operates directly on windows of raw IMU measurements instead of relying on the previous handcrafted-feature pipeline.

Each inference window contains:

```text
128 × 6
```

or:

```text
768 raw sensor values
```

The same sensor-domain normalization statistics calculated from the training data are stored in:

```text
raw_scaler.h
```

These values are reused during embedded inference so that the preprocessing performed on the microcontroller matches the preprocessing used during model development.

---

# Model Input

The model input tensor was verified directly on the physical microcontroller.

Input tensor:

```text
Type:
INT8

Dimensions:
1 × 128 × 6

Elements:
768

Bytes:
768
```

Quantization parameters:

```text
Scale:
0.04553470

Zero point:
13
```

Therefore, each inference processes one complete IMU window:

```text
1 × 128 × 6
```

---

# Model Output

The model output tensor was also verified directly on-device.

Output tensor:

```text
Type:
INT8

Dimensions:
1 × 4

Elements:
4

Bytes:
4
```

Output quantization parameters:

```text
Scale:
0.00390625

Zero point:
-128
```

The four output values correspond to the four gesture classes.

After inference, the output tensor can be dequantized and the class with the highest score selected using `argmax`.

---

# Depthwise CNN

The final embedded neural network is an INT8-quantized Depthwise Convolutional Neural Network.

Depthwise convolutions are useful for TinyML because they can significantly reduce computational complexity compared with standard convolutional layers.

The model was designed to process temporal six-axis IMU signals while remaining small enough to execute efficiently on a resource-constrained microcontroller.

Embedded model size:

```text
10,368 bytes
```

or approximately:

```text
10.1 KB
```

The model is stored inside:

```text
depthwise_model.h
```

as a C byte array.

---

# INT8 Quantization

The neural network was converted to an INT8 TensorFlow Lite model for embedded deployment.

INT8 quantization provides several advantages for microcontroller inference:

- Lower memory consumption
- Smaller model size
- Efficient integer operations
- Reduced inference cost
- Compatibility with TensorFlow Lite Micro

The embedded model therefore uses:

```text
INT8 input
INT8 weights
INT8 intermediate tensors
INT8 output
```

The model input must be quantized according to the input tensor parameters before inference.

Conceptually:

```text
quantized_value =
standardized_value / input_scale
+ input_zero_point
```

The result is then clamped to the valid INT8 range:

```text
-128 ... 127
```

---

# Raw IMU Scaling

Training-time preprocessing statistics are stored in:

```text
raw_scaler.h
```

The current embedded values are:

```text
Axis 0
mean = -0.00392063
std  =  0.25857764

Axis 1
mean = -0.09010717
std  =  0.60423815

Axis 2
mean = -0.13027927
std  =  0.75048846

Axis 3
mean = -1.59155166
std  = 18.72109604

Axis 4
mean =  0.14677145
std  = 17.97825241

Axis 5
mean = -0.04352759
std  = 56.13960266
```

Each raw sensor value is standardized using:

```text
standardized_value =
(raw_value - mean) / std
```

The standardized value is subsequently quantized to INT8 before being copied into the TensorFlow Lite input tensor.

This preprocessing step is critical because the embedded model must receive data in the same statistical domain used during training.

---

# Embedded Deployment

The model is deployed using TensorFlow Lite for Microcontrollers.

Current embedded development stack:

```text
Arduino IDE 2.3.8
Chirale_TensorFlowLite 2.0.0
TensorFlow Lite for Microcontrollers
C / C++
INT8 Depthwise CNN
```

Primary embedded files:

```text
Depthwise_Gesture_Classifier.ino
depthwise_model.h
raw_scaler.h
```

The TensorFlow Lite schema was verified directly on the board:

```text
Model schema version:
3

Runtime schema version:
3
```

Therefore:

```text
Model schema = Runtime schema
```

and the model is compatible with the deployed TensorFlow Lite Micro runtime.

---

# TensorFlow Lite Micro Operators

The model requires eight TensorFlow Lite operators:

```text
ExpandDims
DepthwiseConv2D
Conv2D
Reshape
MaxPool2D
Mean
FullyConnected
Softmax
```

Instead of loading every available TensorFlow Lite operator, only the required operations are registered.

The resolver is therefore created using:

```cpp
tflite::MicroMutableOpResolver<8>
```

and populated with:

```cpp
resolver.AddExpandDims();
resolver.AddDepthwiseConv2D();
resolver.AddConv2D();
resolver.AddReshape();
resolver.AddMaxPool2D();
resolver.AddMean();
resolver.AddFullyConnected();
resolver.AddSoftmax();
```

Explicit operator registration reduces unnecessary runtime overhead and makes the embedded inference configuration easier to inspect and debug.

---

# Tensor Arena

TensorFlow Lite Micro uses a statically allocated tensor arena for runtime tensor storage.

The current configuration uses:

```cpp
constexpr size_t kTensorArenaSize = 64 * 1024;
```

which corresponds to:

```text
65,536 bytes
```

The arena is explicitly aligned:

```cpp
alignas(16)
static uint8_t tensorArena[kTensorArenaSize];
```

Controlled on-device testing measured:

```text
Arena capacity:
65,536 bytes

Arena used:
6,852 bytes
```

Therefore the neural network currently requires only approximately:

```text
6.7 KB
```

of the allocated tensor arena.

This is approximately:

```text
10.5%
```

of the current 64 KB allocation.

The result also proved that the previous `AllocateTensors()` failure was not caused by insufficient tensor-arena capacity.

---

# Deployment Debugging

One of the most important technical challenges in the project occurred during TensorFlow Lite Micro initialization.

The firmware could successfully complete:

```text
Compilation
Model loading
Schema validation
Operator registration
MicroInterpreter construction
```

but execution could unexpectedly stop inside:

```cpp
interpreter.AllocateTensors();
```

The behavior was unusual because the call did not simply return a normal TensorFlow Lite error.

Instead, execution could stop before returning from `AllocateTensors()`.

A controlled diagnostic strategy was therefore introduced.

---

# Controlled Stripping Strategy

The complete application was progressively reduced to isolate the source of the failure.

The debugging process followed:

```text
Full application
      ↓
Remove IMU
      ↓
Remove live capture
      ↓
Remove preprocessing
      ↓
Remove sample buffers
      ↓
Minimal TensorFlow Lite runtime
      ↓
Test model
      ↓
Reintroduce dependencies individually
```

This allowed each embedded subsystem to be tested independently.

The controlled tests eventually demonstrated that the problem was associated with the memory alignment of the embedded TensorFlow Lite FlatBuffer.

---

# Controlled Test 3 — Minimal TFLM Runtime

Controlled Test 3 removed the application-specific components and retained only the minimal TensorFlow Lite Micro inference pipeline.

Pipeline:

```text
GetModel()
     ↓
Schema validation
     ↓
Operator resolver
     ↓
MicroInterpreter
     ↓
AllocateTensors()
     ↓
Deterministic INT8 input
     ↓
Invoke()
```

Result:

```text
GetModel()        PASS
Schema            PASS
Resolver          PASS
Interpreter       PASS
AllocateTensors() PASS
Invoke()          PASS
```

The deterministic input produced:

```text
Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625
```

Predicted class:

```text
Class 2
```

This proved that:

```text
Model
+
TensorFlow Lite Micro
+
Operator resolver
+
Tensor allocation
+
Inference
```

were fundamentally compatible with the target microcontroller.

---

# Controlled Test 4 — raw_scaler Reintroduced

The next experiment restored:

```text
raw_scaler.h
```

while keeping the application otherwise minimal.

The scaler data could be accessed successfully:

```text
raw_scaler reference PASS
```

However, execution again stopped during:

```cpp
AllocateTensors();
```

This initially appeared to indicate that `raw_scaler.h` itself was causing the problem.

Further testing demonstrated that this interpretation was incorrect.

The scaler values were valid.

Instead, adding the scaler changed the compiled binary memory layout.

That change also moved the embedded TensorFlow Lite model to a different memory address.

This observation became the key clue in identifying the actual problem.

---

# Controlled Test 5 — Aligned RAM Model

The address of the original embedded model was inspected.

The model was observed at:

```text
Original model address:
0x330A4
```

Alignment check:

```text
Original model mod16:
4
```

Therefore:

```text
model_address % 16 != 0
```

The model was then copied into an explicitly aligned RAM buffer.

The aligned model address became:

```text
Aligned model address:
0x20011030
```

Alignment check:

```text
Aligned model mod16:
0
```

The aligned RAM copy was then passed to TensorFlow Lite Micro.

Result:

```text
raw_scaler.h       PASS
GetModel()         PASS
Schema             PASS
Resolver           PASS
Interpreter        PASS
AllocateTensors()  PASS
Invoke()           PASS
```

This experiment strongly indicated that model alignment was responsible for the previous runtime instability.

However, the aligned RAM solution duplicated the entire model in SRAM.

Because the model size is:

```text
10,368 bytes
```

this workaround consumed approximately 10 KB of additional dynamic memory.

A better production solution was therefore required.

---

# Controlled Test 6 — Direct Aligned Flash Model

Controlled Test 6 removed the temporary RAM model copy.

Instead, alignment was applied directly to the model definition inside:

```text
depthwise_model.h
```

The model is declared using:

```cpp
alignas(16)
const unsigned char depthwise_model[] = {
    ...
};
```

This ensures that the model FlatBuffer itself begins at a 16-byte-aligned memory address.

No `memcpy()` is required.

No second model buffer is created.

The model is passed directly to:

```cpp
tflite::GetModel(depthwise_model);
```

---

## Test 6 Alignment Verification

Runtime inspection reported:

```text
depthwise_model mod16:
0
```

The Tensor Arena was also aligned:

```text
Tensor arena mod16:
0
```

Therefore both critical embedded memory structures satisfied the expected alignment.

---

## Test 6 Tensor Allocation

The TensorFlow Lite Micro interpreter successfully executed:

```cpp
interpreter.AllocateTensors();
```

Result:

```text
Allocate status:
0

[A3] AllocateTensors SUCCESS
```

The measured arena usage was:

```text
Arena capacity:
65,536 bytes

Arena used:
6,852 bytes
```

---

## Test 6 Tensor Verification

Input tensor:

```text
Type:
INT8

Bytes:
768

Dimensions:
1 × 128 × 6
```

Output tensor:

```text
Type:
INT8

Bytes:
4

Dimensions:
1 × 4
```

Both tensors matched the expected model architecture.

---

## Test 6 Invoke

After filling the input tensor with a deterministic test input, the firmware executed:

```cpp
interpreter.Invoke();
```

Result:

```text
Invoke status:
0

[V3] Invoke SUCCESS
```

This confirmed that the complete inference pipeline could execute using the model directly from its aligned embedded representation.

---

## Test 6 Final Result

The final serial output reported:

```text
depthwise_model alignment : PASS
raw_scaler.h              : PASS
GetModel()                : PASS
Schema                    : PASS
Resolver                  : PASS
Interpreter               : PASS
AllocateTensors()         : PASS
Input/output tensors      : PASS
Invoke()                  : PASS
RAM model copy            : NOT USED
```

Final result:

```text
*** DIRECT ALIGNED FLASH MODEL TEST PASSED ***
```

This represents the current major deployment milestone of the project.

---

# Root Cause

The controlled debugging sequence strongly demonstrated that the previous TensorFlow Lite Micro instability was associated with memory alignment of the embedded model.

The critical comparison was:

```text
Original embedded model
Address mod16 = 4
        ↓
AllocateTensors instability
```

compared with:

```text
Aligned model
Address mod16 = 0
        ↓
AllocateTensors PASS
        ↓
Invoke PASS
```

The final solution was therefore to explicitly align the model itself:

```cpp
alignas(16)
const unsigned char depthwise_model[] = {
    ...
};
```

This is preferable to the temporary Test 5 workaround because the model remains in its embedded program-memory representation and does not need to be duplicated in SRAM.

---

# Memory Usage

The successful Test 6 firmware compiled with approximately:

```text
Program storage:
183,352 bytes / 983,040 bytes
```

which corresponds to approximately:

```text
18%
```

of the available program storage.

Dynamic memory:

```text
112,104 bytes / 262,144 bytes
```

which corresponds to approximately:

```text
42%
```

of available dynamic memory.

Remaining dynamic memory:

```text
150,040 bytes
```

This leaves substantial memory headroom for restoring:

```text
IMU acquisition
real-time buffering
preprocessing
quantization
classification logic
```

---

# Memory Efficiency

The Test 5 RAM workaround required an additional copy of the model.

Model size:

```text
10,368 bytes
```

The final Test 6 configuration removes this duplicate.

Therefore approximately:

```text
10 KB
```

of SRAM is recovered compared with the aligned-RAM workaround.

The current model itself remains stored in its embedded aligned representation.

The Tensor Arena currently consumes:

```text
6,852 bytes
```

during inference.

---

# Inference Performance

Controlled Test 6 measured on-device inference latency of approximately:

```text
20.19 ms
```

for one complete:

```text
128 × 6
```

INT8 input window.

Considering only neural-network computation, this corresponds theoretically to approximately:

```text
~49 model invocations per second
```

However, the final real-time classification rate will also depend on:

- IMU sample rate
- Window acquisition time
- Trigger logic
- Preprocessing
- INT8 quantization
- Serial output
- Application-level cooldown logic

Therefore the final gesture-recognition latency will be evaluated separately after real-time IMU integration.

---

# Deterministic Test Output

Controlled Test 6 used a synthetic deterministic input rather than an actual gesture.

The model produced:

```text
Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625
```

Predicted class:

```text
Class 2
```

Confidence:

```text
≈ 0.512
```

This result should not be interpreted as gesture-classification accuracy.

The purpose of this test was exclusively to verify:

```text
input tensor
      ↓
CNN
      ↓
Invoke()
      ↓
output tensor
```

on the physical microcontroller.

---

# Confirmed Results

The following components have now been experimentally verified on the physical microcontroller:

- Dataset successfully collected from IMU measurements.
- Four gesture categories defined.
- Depthwise CNN trained.
- INT8 model generated.
- Embedded model exported.
- Model size verified as 10,368 bytes.
- `raw_scaler.h` generated.
- TensorFlow Lite Micro integrated.
- TensorFlow Lite schema version verified.
- Required operators identified.
- Required operators registered.
- `MicroInterpreter` successfully constructed.
- Tensor Arena successfully allocated.
- Tensor Arena alignment verified.
- Model alignment verified.
- Input tensor successfully allocated.
- Output tensor successfully allocated.
- Input tensor verified as `1 × 128 × 6`.
- Output tensor verified as `1 × 4`.
- Input quantization parameters verified.
- Output quantization parameters verified.
- `raw_scaler.h` successfully linked.
- `AllocateTensors()` successfully executed.
- `Invoke()` successfully executed.
- Four-class model output generated.
- Model successfully executed directly from aligned embedded memory.
- Temporary RAM model copy removed.
- On-device inference latency measured at approximately 20 ms.
- Firmware successfully compiled.
- Firmware successfully uploaded to the physical board.
- Controlled Test 6 successfully completed.

---

# Current Project Status

Current development status:

```text
Dataset collection                     DONE
Gesture labels                         DONE

Depthwise CNN training                 DONE
INT8 quantization                      DONE
Model export                           DONE
Embedded model generation              DONE
raw_scaler generation                  DONE

TensorFlow Lite Micro integration      DONE
Operator resolver                      DONE
Schema validation                      PASS
Interpreter construction               PASS
Tensor allocation                      PASS
Inference                              PASS

Controlled Test 3                      PASS
Controlled Test 4                      DIAGNOSTIC FAILURE
Controlled Test 5                      PASS
Controlled Test 6                      PASS

Model alignment issue                  IDENTIFIED
16-byte model alignment                IMPLEMENTED
Direct aligned model execution         PASS
RAM model duplication                  REMOVED

Input tensor validation                PASS
Output tensor validation               PASS
INT8 inference                         PASS
Inference latency measurement          DONE

Live IMU integration                   IN PROGRESS
128 × 6 real-time capture              PENDING
Real-time standardization              PENDING
Real-time INT8 quantization            PENDING
Real gesture inference                 PENDING
Gesture label mapping                  PENDING
Confidence reporting                   PENDING
Real-world accuracy validation         PENDING
Final performance evaluation           PENDING
```

---

# Current Milestone

## Direct Aligned TFLite Micro Inference — PASSED

The current verified embedded configuration is:

```text
Model:
INT8 Depthwise CNN

Model size:
10,368 bytes

Input:
1 × 128 × 6 INT8

Output:
1 × 4 INT8

Tensor Arena:
65,536 bytes

Observed Tensor Arena usage:
6,852 bytes

Model alignment:
16-byte aligned

raw_scaler.h:
PRESENT

RAM model copy:
NOT USED

AllocateTensors():
PASS

Invoke():
PASS

Inference latency:
~20 ms
```

The core TinyML deployment stage is therefore operational.

---

# Next Development Stage

The next stage is to connect the already validated neural-network runtime to the live IMU stream.

The target real-time pipeline is:

```text
On-board IMU
     ↓
Read:
ax ay az gx gy gz
     ↓
Capture 128 × 6 window
     ↓
Apply:
raw_mean / raw_std
     ↓
Quantize to INT8
     ↓
Fill TFLM input tensor
     ↓
Invoke Depthwise CNN
     ↓
Read 4 output scores
     ↓
Dequantize
     ↓
Argmax
     ↓
Gesture
     ↓
Confidence
```

Final gesture output:

```text
circle
left_right
rest
up_down
```

The important point is that the next development stage does **not** require rebuilding the TensorFlow Lite Micro deployment pipeline.

The following components are already validated:

```text
Model
Model alignment
Schema
Operator resolver
MicroInterpreter
Tensor Arena
Input tensor
Output tensor
INT8 inference
Invoke()
Memory capacity
Inference latency
```

The remaining work primarily concerns real-time sensor integration and validation.

---

# Real-Time Integration Goals

The next implementation will need to reproduce the same sensor-data conditions used during dataset collection.

Important factors include:

```text
IMU sensor source
Channel ordering
Sensor units
Sample rate
Window size
Preprocessing
Gesture timing
```

The channel order must remain:

```text
ax
ay
az
gx
gy
gz
```

The window must remain:

```text
128 × 6
```

and the training statistics stored in:

```text
raw_scaler.h
```

must be applied before INT8 quantization.

This ensures that live sensor measurements are presented to the neural network in the same format used during training.

---

# Repository Structure

A simplified repository structure is:

```text
Gesture-Classification/
│
├── data/
│   └── gesture dataset
│
├── training/
│   └── model training and evaluation
│
├── models/
│   └── quantized TFLite model
│
├── arduino/
│   │
│   ├── Depthwise_Gesture_Classifier.ino
│   ├── depthwise_model.h
│   └── raw_scaler.h
│
└── README.md
```

The exact repository organization may evolve as the real-time classifier and final evaluation files are added.

---

# Development Strategy

A central engineering principle of this project has been:

> Validate each embedded subsystem independently before integrating the complete real-time classifier.

Instead of debugging all components simultaneously:

```text
IMU
+
buffering
+
preprocessing
+
quantization
+
TensorFlow Lite Micro
+
memory
+
serial communication
```

the project progressively isolated the inference system.

This strategy allowed the deployment problem to be narrowed from a general:

```text
AllocateTensors() failure
```

to a specific model memory-alignment issue.

The sequence:

```text
Minimal TFLM
     ↓
Scaler reintroduced
     ↓
Memory addresses inspected
     ↓
Aligned RAM experiment
     ↓
Direct aligned model
```

provided experimental evidence for the final solution.

This debugging process is also an important result of the project because it demonstrates a systematic methodology for diagnosing embedded TinyML deployment failures.

---

# Key Engineering Lessons

Several practical lessons were obtained during deployment.

### 1. Successful compilation does not guarantee runtime correctness

The firmware compiled successfully even when runtime inference initialization failed.

Embedded ML systems therefore require on-device runtime validation.

### 2. Tensor Arena size was not the actual problem

A 64 KB arena was available, while actual model usage was only approximately:

```text
6.85 KB
```

Increasing the arena alone would therefore not have solved the failure.

### 3. Binary layout can affect runtime behavior

Adding apparently unrelated static data changed the location of the embedded model.

This changed runtime behavior even though the model bytes themselves were unchanged.

### 4. Memory alignment matters

The critical difference was:

```text
mod16 = 4
```

versus:

```text
mod16 = 0
```

Explicit model alignment solved the runtime instability.

### 5. Controlled stripping is effective for embedded debugging

Reducing the application to a minimal reproducible configuration made it possible to separate:

```text
model problems
runtime problems
memory problems
preprocessing problems
application problems
```

### 6. Diagnostic workarounds should not automatically become production solutions

The aligned RAM copy was useful because it demonstrated the root cause.

However, direct alignment of the embedded model was a better final solution because it eliminated unnecessary SRAM duplication.

---

# Final Target

The final project will perform gesture classification completely on-device:

```text
User performs gesture
        ↓
Arduino IMU records movement
        ↓
128 × 6 temporal window
        ↓
Embedded preprocessing
        ↓
INT8 Depthwise CNN
        ↓
TensorFlow Lite Micro inference
        ↓
Gesture prediction
        ↓
Confidence score
```

No external computer should be required for inference after deployment.

The target gesture classes are:

```text
circle
left_right
rest
up_down
```

---

# About

TinyML gesture-classification project deploying a quantized INT8 Depthwise CNN on an Arduino Nano 33 BLE platform using six-axis IMU data, embedded preprocessing, TensorFlow Lite Micro, direct aligned model execution, and real-time on-device inference.

The project includes the complete workflow from IMU dataset collection and neural-network development to quantization, embedded deployment, low-level memory debugging, and real-time gesture-recognition integration.

---

# Project Status

**Active development — core TinyML inference deployment is operational.**

The INT8 Depthwise CNN now executes successfully on the physical microcontroller using a directly aligned embedded model.

The following critical deployment stages have been completed:

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

The temporary aligned RAM model used during debugging is no longer required.

The current development stage is:

```text
Live IMU
   ↓
128 × 6 acquisition
   ↓
Preprocessing
   ↓
INT8 quantization
   ↓
Depthwise CNN
   ↓
circle / left_right / rest / up_down
```

The next milestone is successful real-time classification of physical gestures using live IMU measurements.
