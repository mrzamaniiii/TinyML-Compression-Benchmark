# TinyML Gesture Classification on Arduino Nano 33 BLE

Real-time IMU gesture classification on the Arduino Nano 33 BLE using an INT8 Depthwise CNN and TensorFlow Lite for Microcontrollers.

This project investigates the complete deployment pipeline of a quantized neural network on a resource-constrained microcontroller, including IMU acquisition, preprocessing, INT8 inference, memory debugging, TensorFlow Lite Micro integration, and embedded model alignment.

---

## Project Overview

The goal of this project is to deploy a lightweight gesture-classification neural network directly on an:

```text
Arduino Nano 33 BLE
```

The intended final pipeline is:

```text
Arduino Nano 33 BLE IMU
        ↓
Motion Detection
        ↓
128 × 6 IMU Window
        ↓
Raw Feature Scaling
        ↓
INT8 Quantization
        ↓
Depthwise CNN
        ↓
TensorFlow Lite Micro
        ↓
4-Class Output
        ↓
Gesture + Confidence
```

The model processes six IMU channels over a window of 128 samples:

```text
Input shape: 1 × 128 × 6
```

and produces four output probabilities:

```text
Output shape: 1 × 4
```

---

# Hardware

Target board:

```text
Arduino Nano 33 BLE
```

The board provides:

- ARM Cortex-M4F microcontroller
- 1 MB Flash
- 256 KB SRAM
- On-board IMU
- USB serial interface

The project is designed to perform inference entirely on-device without requiring a PC or cloud service during classification.

---

# Software Stack

Development and deployment use:

```text
Arduino IDE 2.3.8
Arduino Nano 33 BLE
TensorFlow Lite for Microcontrollers
INT8 quantized CNN
C / C++
```

The primary project files are:

```text
Depthwise_Gesture_Classifier.ino
depthwise_model.h
raw_scaler.h
```

---

# Neural Network

The deployed model is an INT8 quantized Depthwise CNN.

Embedded model size:

```text
10,368 bytes
```

TensorFlow Lite schema:

```text
Model schema version:   3
Runtime schema version: 3
```

The model uses eight operator codes.

The required TensorFlow Lite Micro operators were identified as:

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

These operators are registered explicitly using a `MicroMutableOpResolver`.

---

# Input Tensor

The model input tensor has:

```text
Type: INT8
Dimensions: 1 × 128 × 6
Elements: 768
Bytes: 768

Scale:      0.04553470
Zero point: 13
```

Therefore, one inference window contains:

```text
128 samples × 6 IMU channels
```

---

# Output Tensor

The output tensor has:

```text
Type: INT8
Dimensions: 1 × 4
Elements: 4
Bytes: 4

Scale:      0.00390625
Zero point: -128
```

The four values correspond to the four gesture classes.

---

# Raw Feature Scaling

The project includes:

```text
raw_scaler.h
```

containing the preprocessing statistics used during training.

Observed values:

```text
Axis 0 mean = -0.00392063    std = 0.25857764
Axis 1 mean = -0.09010717    std = 0.60423815
Axis 2 mean = -0.13027927    std = 0.75048846
Axis 3 mean = -1.59155166    std = 18.72109604
Axis 4 mean =  0.14677145    std = 17.97825241
Axis 5 mean = -0.04352759    std = 56.13960266
```

These statistics will eventually be used in the live preprocessing pipeline before INT8 quantization.

---

# Tensor Arena

A 64 KB TensorFlow Lite Micro tensor arena was used during debugging:

```cpp
constexpr size_t kTensorArenaSize = 64 * 1024;
```

The arena was explicitly aligned.

During the successful controlled test, TensorFlow Lite Micro reported approximately:

```text
Arena capacity: 65536 bytes
Arena used:      6852 bytes
```

Therefore, the previous `AllocateTensors()` failure was **not caused by an undersized tensor arena**.

---

# Debugging Background

During integration of the complete classifier, execution repeatedly stopped during:

```cpp
interpreter.AllocateTensors();
```

The failure was difficult to diagnose because:

- the sketch compiled successfully,
- the model schema was valid,
- all required operators were registered,
- the interpreter constructor completed,
- the board had sufficient RAM,
- and no ordinary TensorFlow Lite error was returned.

Instead, execution could stop inside `AllocateTensors()`.

A controlled stripping strategy was therefore used.

---

# Controlled Stripping Method

The application was progressively reduced until only the TensorFlow Lite Micro runtime remained.

The debugging procedure was:

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
Remove classifier globals
      ↓
Test minimal TFLM runtime
      ↓
Reintroduce dependencies one by one
```

This allowed each subsystem to be tested independently.

---

# Controlled Stripping Test 3

Test 3 was an absolute-minimal TensorFlow Lite Micro test.

It contained no:

```text
IMU
raw_scaler
sampleBuffer
motion detection
gesture capture
preprocessing
live classifier logic
```

The test performed:

```text
GetModel()
    ↓
Schema validation
    ↓
Resolver creation
    ↓
MicroInterpreter construction
    ↓
AllocateTensors()
    ↓
Input/output tensor inspection
    ↓
Deterministic INT8 input
    ↓
Invoke()
    ↓
Output inspection
```

The input tensor was filled with its quantized zero point:

```text
13
```

so that inference could be tested independently of real sensor data.

## Test 3 Result

Tensor allocation succeeded:

```text
[A1] ENTER AllocateTensors()
[A2] EXIT AllocateTensors()

Allocate status: 0

[A3] AllocateTensors SUCCESS
```

The tensors were correctly created:

```text
Input:
1 × 128 × 6
768 bytes

Output:
1 × 4
4 bytes
```

Inference also succeeded:

```text
[V1] ENTER Invoke()
[V2] EXIT Invoke()

Invoke status: 0

[V3] Invoke SUCCESS
```

Observed output:

```text
Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625
```

The test concluded with:

```text
*** MINIMAL TFLM TEST PASSED ***
```

This proved that the model and TensorFlow Lite Micro runtime were fundamentally functional on the Arduino Nano 33 BLE.

---

# Controlled Stripping Test 4

Test 4 started from the successful Test 3 configuration and added only:

```text
raw_scaler.h
```

No IMU code, sample buffer, motion detector, or live classifier was restored.

The scaler arrays were deliberately accessed at runtime.

The scaler test succeeded:

```text
[S1] ENTER raw_scaler test

...

Scaler probe: 92.73963928

[S2] raw_scaler reference SUCCESS
```

The model was successfully parsed.

The schema was valid.

The resolver was successfully created.

The interpreter constructor also completed.

However, execution stopped at:

```text
[A1] ENTER AllocateTensors()
```

without reaching:

```text
[A2] EXIT AllocateTensors()
```

This was an important observation.

---

# Test 4 Hypothesis

Initially, `raw_scaler.h` appeared to be responsible for the problem.

However, its data did not explain a significant SRAM increase.

The evidence instead suggested that introducing `raw_scaler.h` changed the layout of the compiled binary.

This could change the address of:

```text
depthwise_model
```

in program memory.

The next test therefore investigated model alignment directly.

---

# Controlled Stripping Test 5

Test 5 retained:

```text
raw_scaler.h
```

exactly as in Test 4.

The important change was that the entire TFLite model was copied into an explicitly 16-byte-aligned RAM buffer before creating the TFLite model object.

Conceptually:

```text
depthwise_model
      ↓
memcpy()
      ↓
16-byte aligned RAM buffer
      ↓
GetModel()
      ↓
MicroInterpreter
```

---

# Model Alignment Discovery

The original embedded model address was inspected.

Observed:

```text
Original model addr: 0x330A4
Original model mod16: 4
```

Therefore:

```text
Original model address % 16 = 4
```

The model was **not 16-byte aligned**.

The RAM model copy was:

```text
Aligned model addr: 0x20011030
Aligned model mod16: 0
```

Therefore:

```text
Aligned model address % 16 = 0
```

The tensor arena was also aligned:

```text
Tensor arena addr: 0x20001030
Tensor arena mod16: 0
```

---

# Test 5 Model Copy Verification

The model was copied to aligned RAM and verified before use.

Serial output confirmed:

```text
[C1] ENTER model memcpy
[C2] Model copied to aligned RAM
[C3] Model RAM copy verified
```

The same model bytes were therefore being executed.

The important difference was their memory placement.

---

# Test 5 Tensor Allocation

After loading the model from the aligned RAM buffer:

```text
[A1] ENTER AllocateTensors()
[A2] EXIT AllocateTensors()

Allocate status: 0

[A3] AllocateTensors SUCCESS
```

TensorFlow Lite Micro reported:

```text
Arena used: 6852
```

This is far below the available:

```text
65536 bytes
```

and confirms that tensor-arena capacity was not the source of the previous failure.

---

# Test 5 Inference

Input and output tensors were successfully obtained:

```text
Input bytes:       768
Input scale:       0.04553470
Input zero point:  13

Output bytes:      4
Output scale:      0.00390625
Output zero point: -128
```

The test input was filled successfully:

```text
[F1] Filling input tensor
[F2] Input filled
```

Inference then completed:

```text
[V1] ENTER Invoke()
[V2] EXIT Invoke()

Invoke status: 0

[V3] Invoke SUCCESS
```

---

# Test 5 Output

The model produced:

```text
Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625
```

The complete test ended with:

```text
CONTROLLED STRIPPING TEST 5 PASSED
```

and:

```text
raw_scaler.h: PRESENT
Model source: 16-byte aligned RAM copy
AllocateTensors(): PASS
Invoke(): PASS
```

---

# Root-Cause Finding

The controlled tests produced the following important comparison:

```text
TEST 3
--------------------------------
Minimal TFLM
AllocateTensors()    PASS
Invoke()             PASS


TEST 4
--------------------------------
Test 3 + raw_scaler.h
AllocateTensors()    FAIL / STOP


TEST 5
--------------------------------
raw_scaler.h present
same TFLite model
model copied to aligned RAM
AllocateTensors()    PASS
Invoke()             PASS
```

The original model placement observed during Test 5 was:

```text
mod16 = 4
```

while the successful RAM copy was:

```text
mod16 = 0
```

The current evidence therefore strongly indicates that the previous runtime failure was associated with the **memory placement/alignment of the embedded TFLite FlatBuffer**.

---

# What Has Been Ruled Out

The controlled tests provide strong evidence against several earlier possibilities.

## Model corruption

Unlikely.

The same model successfully executes when copied to aligned RAM.

## Invalid TFLite schema

Ruled out.

```text
Model schema version:   3
Runtime schema version: 3
```

## Missing operators

Ruled out.

All eight required operators are successfully registered.

## MicroInterpreter construction

Ruled out.

Construction completes successfully.

## Insufficient tensor arena

Ruled out.

Observed usage:

```text
6852 / 65536 bytes
```

## raw_scaler values themselves

Not the direct cause.

The scaler is present and actively referenced during successful Test 5 inference.

## IMU subsystem

Not required to reproduce the problem.

The failure was reproduced without IMU initialization or sensor capture.

---

# Current Working Solution

The currently proven workaround is to copy the embedded model into an aligned RAM buffer.

Conceptually:

```cpp
alignas(16)
static uint8_t alignedModel[depthwise_model_len];

memcpy(
    alignedModel,
    depthwise_model,
    depthwise_model_len
);

const tflite::Model* model =
    tflite::GetModel(alignedModel);
```

With this configuration:

```text
raw_scaler.h       PASS
GetModel()         PASS
Schema             PASS
Resolver           PASS
Interpreter        PASS
AllocateTensors()  PASS
Input tensor       PASS
Output tensor      PASS
Invoke()           PASS
```

---

# Limitation of the RAM-Copy Workaround

Although the RAM-copy solution works, it is not ideal for the final firmware.

The model size is:

```text
10,368 bytes
```

Copying it to SRAM therefore consumes approximately another 10 KB of RAM.

The final production implementation should avoid this unnecessary duplication.

---

# Preferred Production Fix

The preferred solution is to align the model array itself when generating or declaring:

```text
depthwise_model
```

For example:

```cpp
alignas(16)
const unsigned char depthwise_model[] = {
    ...
};
```

or with an appropriate compiler-specific alignment attribute.

The model should then remain in program memory while satisfying the required alignment.

The address should be checked at runtime:

```cpp
Serial.println(
    (uintptr_t)depthwise_model % 16
);
```

Expected result:

```text
0
```

After that, the RAM-copy workaround can be removed and the following operations retested:

```text
GetModel()
AllocateTensors()
Invoke()
```

---

# Serial Checkpoint Strategy

Explicit checkpoints were added around critical TensorFlow Lite operations.

Example:

```cpp
checkpoint("[A1] ENTER AllocateTensors()");

TfLiteStatus alloc_status =
    interpreter.AllocateTensors();

checkpoint("[A2] EXIT AllocateTensors()");
```

This made it possible to distinguish between:

```text
Function returns an error
```

and:

```text
Execution stops inside function
```

Similar checkpoints were used around:

```text
GetModel()
Resolver construction
MicroInterpreter construction
AllocateTensors()
Tensor access
Invoke()
```

This approach was essential for locating the runtime failure.

---

# Arduino Upload / COM Port Issue

A separate USB/serial issue was observed during development.

Arduino IDE occasionally reported:

```text
No device found on COM19
```

or:

```text
Failed uploading:
uploading error: exit status 1
```

This problem is independent of TensorFlow Lite Micro.

Successful uploads were confirmed with:

```text
100% (45/45 pages)

Done in ~7–8 seconds
```

When necessary, the Nano 33 BLE bootloader could be entered by double-pressing reset and selecting the newly detected COM port.

---

# Memory Observations

Typical compilation reports during the controlled tests were approximately:

```text
Program storage:
~18%

Dynamic memory:
~42–46%
```

For example, one configuration reported:

```text
Sketch uses 181808 bytes
Global variables use 122472 bytes
```

out of:

```text
983040 bytes program storage
262144 bytes dynamic memory
```

Therefore, the project still has substantial Flash and SRAM headroom.

---

# Confirmed Findings

At the current stage, the following have been demonstrated experimentally on the Arduino Nano 33 BLE:

- The firmware compiles successfully.
- The INT8 Depthwise CNN is embedded successfully.
- The model size is 10,368 bytes.
- Model schema version 3 is correct.
- Runtime schema version 3 matches the model.
- All required TFLite operators are known.
- All required operators can be registered.
- `MicroInterpreter` construction succeeds.
- A 64 KB tensor arena is sufficient.
- Actual observed tensor-arena usage is approximately 6,852 bytes.
- Input tensor creation succeeds.
- Output tensor creation succeeds.
- INT8 quantization parameters are accessible.
- `AllocateTensors()` succeeds in the controlled configuration.
- `Invoke()` succeeds on the target MCU.
- The model generates valid four-class output.
- `raw_scaler.h` can coexist with successful inference.
- The same model works when copied to 16-byte-aligned RAM.
- The original model placement observed in Test 5 was not 16-byte aligned.
- The aligned RAM model placement was 16-byte aligned.
- The previous failure is not explained by tensor-arena exhaustion.
- The previous failure does not require the IMU subsystem to occur.

---

# Current Leading Conclusion

The strongest current conclusion is:

> The embedded TFLite FlatBuffer must be placed with suitable memory alignment. Changes elsewhere in the firmware can alter binary layout and expose an alignment-sensitive failure during TensorFlow Lite Micro tensor allocation.

The 16-byte-aligned RAM copy is a proven workaround.

The preferred production solution is to guarantee suitable alignment directly on the embedded model array.

---

# Current Project Status

```text
Model training/conversion              DONE
INT8 quantization                      DONE
Model export                           DONE
Model embedded in firmware             DONE

Model schema validation                PASS
Operator identification                PASS
Resolver construction                  PASS
Interpreter construction               PASS

Controlled Stripping Test 1            DONE
Controlled Stripping Test 2            DONE
Controlled Stripping Test 3            PASS
Controlled Stripping Test 4            FAIL at AllocateTensors()
Controlled Stripping Test 5            PASS

Minimal TFLM inference                 PASS
AllocateTensors()                      PASS
Invoke()                               PASS
raw_scaler + TFLM                      PASS
Tensor arena sizing                    VERIFIED
Model alignment issue                  STRONGLY SUPPORTED

Aligned model in program memory        NEXT
Live IMU restoration                   PENDING
Real gesture inference                 PENDING
Accuracy validation                    PENDING
Performance benchmarking               PENDING
```

---

# Next Development Stage

The debugging phase has now reached an important milestone.

The next goal is no longer to prove that TensorFlow Lite Micro can run the model.

That has already been demonstrated.

The next sequence is:

```text
Explicitly align depthwise_model
        ↓
Keep model in program memory
        ↓
Remove temporary RAM model copy
        ↓
Confirm AllocateTensors()
        ↓
Confirm Invoke()
        ↓
Restore preprocessing
        ↓
Restore sample buffer
        ↓
Restore IMU
        ↓
Capture 128 × 6 windows
        ↓
Scale real IMU data
        ↓
Quantize to INT8
        ↓
Run CNN
        ↓
Dequantize output
        ↓
Select highest-probability class
        ↓
Report gesture + confidence
```

---

# Intended Final Runtime Pipeline

The final classifier should operate as:

```text
              Arduino Nano 33 BLE
                       │
                       ▼
                     IMU
                       │
                       ▼
               Motion Detection
                       │
                       ▼
                128 × 6 Window
                       │
                       ▼
               Feature Scaling
              raw_mean / raw_std
                       │
                       ▼
               INT8 Quantization
                       │
                       ▼
                 Depthwise CNN
                       │
                       ▼
           TensorFlow Lite Micro
                       │
                       ▼
                 4-Class Output
                       │
                       ▼
                 Dequantization
                       │
                       ▼
                     Argmax
                       │
                       ▼
             Gesture + Confidence
```

---

# Development Principle

A central principle of this project is:

> Prove each embedded subsystem independently before rebuilding the complete real-time classifier.

Instead of debugging:

```text
IMU
+
buffering
+
preprocessing
+
quantization
+
TensorFlow Lite
+
memory
+
USB serial
```

simultaneously, each subsystem is introduced in a controlled manner.

This approach successfully isolated an issue that initially appeared to be caused by TensorFlow Lite memory allocation but was strongly associated with the placement/alignment of the embedded model.

---

# Current Milestone

## Minimal On-Device TFLite Micro Inference — PASSED

Current verified configuration:

```text
Platform:
Arduino Nano 33 BLE

Model:
INT8 Depthwise CNN

Model size:
10,368 bytes

Input:
1 × 128 × 6 INT8

Output:
1 × 4 INT8

Tensor arena:
65,536 bytes

Observed arena usage:
6,852 bytes

raw_scaler.h:
PRESENT

Model source:
16-byte-aligned RAM copy

AllocateTensors():
PASS

Invoke():
PASS
```

The next milestone is:

```text
Aligned embedded model
without RAM duplication
        ↓
Full real-time IMU gesture classification
```

---

# About

TinyML gesture-classification project deploying a quantized INT8 Depthwise CNN on the Arduino Nano 33 BLE using TensorFlow Lite Micro, with real-time IMU inference, embedded preprocessing, and memory/alignment optimization.

---

# Project Status

**Active development — core on-device neural-network inference is working.**

The TensorFlow Lite Micro execution path has been successfully validated on the physical Arduino Nano 33 BLE.

The current focus is transitioning from the diagnostic aligned-RAM model configuration to a clean aligned embedded-model implementation, followed by restoration of the complete real-time IMU gesture-classification pipeline.
