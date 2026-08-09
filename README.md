# TinyML Depthwise CNN Gesture Classifier on Arduino Nano 33 BLE

An embedded TinyML project for real-time gesture classification on the **Arduino Nano 33 BLE** using IMU data and a quantized **INT8 Depthwise Convolutional Neural Network (CNN)** deployed with **TensorFlow Lite Micro**.

The project is currently in the **deployment and low-level debugging phase**. The trained model has been successfully converted to a TensorFlow Lite model and embedded into the Arduino firmware. The current investigation focuses on the TensorFlow Lite Micro tensor allocation stage on the target microcontroller.

---

## Project Overview

The goal of this project is to deploy a lightweight neural network for gesture recognition directly on a microcontroller.

The intended inference pipeline is:

```text
IMU
 ↓
Motion Detection
 ↓
Gesture Capture
 ↓
Preprocessing / Scaling
 ↓
INT8 Quantization
 ↓
Depthwise CNN
 ↓
TensorFlow Lite Micro
 ↓
Gesture Probabilities
 ↓
Predicted Gesture
```

The final system is intended to perform the complete classification pipeline locally on the Arduino Nano 33 BLE without requiring cloud inference or a connected computer.

---

## Hardware

Target board:

- Arduino Nano 33 BLE
- ARM Cortex-M4F
- On-board IMU
- 256 KB SRAM
- 1 MB Flash

Development is currently performed through the Arduino IDE using USB serial communication for diagnostics.

---

## Software Environment

Current development environment:

```text
Arduino IDE: 2.3.8
Target: Arduino Nano 33 BLE
Serial baud rate: 115200
```

TensorFlow Lite Micro is integrated through:

```cpp
#include <Chirale_TensorFlowLite.h>
```

with the required TensorFlow Lite Micro headers.

---

## Neural Network

The deployed model is a quantized:

```text
Depthwise CNN
INT8
```

Current embedded model size:

```text
10,368 bytes
```

The model uses TensorFlow Lite schema version:

```text
Model schema version:   3
Runtime schema version: 3
```

Therefore, no model/runtime schema mismatch has been observed.

---

## TensorFlow Lite Operators

Inspection of the embedded model showed that it requires eight TensorFlow Lite operators.

The current resolver contains:

```cpp
tflite::MicroMutableOpResolver<8>
```

with:

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

The corresponding model operator codes observed during diagnostics were:

```text
Opcode 0: builtin=70, version=1
Opcode 1: builtin=4,  version=3
Opcode 2: builtin=3,  version=3
Opcode 3: builtin=22, version=1
Opcode 4: builtin=17, version=2
Opcode 5: builtin=40, version=2
Opcode 6: builtin=9,  version=4
Opcode 7: builtin=25, version=2
```

All required operators can be successfully registered in the `MicroMutableOpResolver`.

---

## Tensor Arena

The current diagnostic configuration uses a 64 KB tensor arena:

```cpp
constexpr int TENSOR_ARENA_SIZE = 64 * 1024;

alignas(16)
static uint8_t tensorArena[TENSOR_ARENA_SIZE];
```

Arena capacity:

```text
65,536 bytes
```

Explicit alignment is used to eliminate obvious tensor-arena alignment problems during testing.

---

# Original Runtime Architecture

The intended full firmware contains several major components.

### 1. IMU acquisition

The Arduino's IMU provides motion data used to capture gestures.

### 2. Motion detection

A configurable motion threshold determines when a gesture capture should begin.

A diagnostic configuration used:

```text
Motion threshold: 0.2000
```

### 3. Sample buffer

Captured IMU samples are stored before preprocessing.

One observed configuration reported:

```text
Sample buffer bytes: 3072
```

### 4. Raw scaler

Training-time scaling parameters are embedded into the firmware through:

```text
raw_scaler.h
```

These parameters are intended to reproduce the preprocessing used during model training.

### 5. INT8 quantization

Preprocessed samples are converted into the quantized representation expected by the TensorFlow Lite model.

### 6. TensorFlow Lite Micro inference

The intended inference sequence is:

```cpp
GetModel()
     ↓
Create Resolver
     ↓
Create MicroInterpreter
     ↓
AllocateTensors()
     ↓
Get Input Tensor
     ↓
Fill Input
     ↓
Invoke()
     ↓
Read Output
```

---

# Project Files

The current Arduino project contains the following important files:

```text
Depthwise_Gesture_Classifier.ino
raw_scaler.h
depthwise_model.h
```

### `Depthwise_Gesture_Classifier.ino`

Main firmware and diagnostic program.

Depending on the debugging stage, this file contains either the complete live gesture-classification pipeline or a stripped TensorFlow Lite Micro diagnostic test.

### `raw_scaler.h`

Contains preprocessing/scaling parameters used by the original classifier.

This dependency has also been deliberately removed in controlled tests to determine whether it contributes to the current runtime failure.

### `depthwise_model.h`

Contains the embedded TensorFlow Lite INT8 model as a C/C++ byte array.

Current model size:

```text
10,368 bytes
```

---

# Current Debugging Investigation

During integration of the complete live classifier, execution repeatedly stopped during:

```cpp
interpreter.AllocateTensors();
```

To locate the failure precisely, extensive serial checkpoints were added.

For example:

```text
[D1] Loading model
[D2] GetModel returned
[D3] Schema OK

[R0] Creating resolver
...
[R9] Resolver ready

[I1] Creating interpreter
[I2] Interpreter created

[A1] ENTER AllocateTensors()
```

The critical observation was that execution reached:

```text
[A1] ENTER AllocateTensors()
```

but did not reach the checkpoint immediately after `AllocateTensors()`.

This strongly localizes the current failure to execution occurring during tensor allocation.

---

# Controlled Stripping Tests

Rather than changing multiple components simultaneously, the firmware has been progressively stripped down to isolate the failing subsystem.

---

## Controlled Stripping Test 1

The first stripped configuration removed major parts of the live classification pipeline while retaining enough of the original application structure to test TensorFlow Lite initialization.

The model was successfully:

- located in memory,
- parsed,
- schema-validated,
- registered with the required operators,
- passed to the interpreter.

Execution nevertheless stopped during:

```cpp
AllocateTensors()
```

This suggested that the failure was not simply occurring during model loading or resolver construction.

---

## Controlled Stripping Test 2

The second test removed additional classifier-specific dependencies.

The Serial Monitor explicitly confirmed:

```text
CONTROLLED STRIPPING TEST 2

NO IMU dependency
NO raw_scaler
NO sampleBuffer
NO live classifier globals
```

The initialization sequence continued successfully through:

```text
[D1] Loading model
[D2] GetModel returned
[D3] Schema OK
```

followed by successful operator registration:

```text
[R0] Creating resolver
[R1] AddExpandDims
[R2] AddDepthwiseConv2D
[R3] AddConv2D
[R4] AddReshape
[R5] AddMaxPool2D
[R6] AddMean
[R7] AddFullyConnected
[R8] AddSoftmax
[R9] Resolver ready
```

and successful interpreter construction:

```text
[I1] Creating interpreter
[I2] Interpreter created
```

The final observed checkpoint was again:

```text
[A1] ENTER AllocateTensors()
```

Therefore, removing the following components did not eliminate the problem:

```text
IMU dependency
raw_scaler
sampleBuffer
live classifier globals
```

This significantly narrowed the investigation.

---

# Controlled Stripping Test 3

The project has now reached **Controlled Stripping Test 3**.

This is an absolute-minimal TensorFlow Lite Micro test.

The firmware contains no:

```text
IMU
raw_scaler
sampleBuffer
motion detection
gesture capture
preprocessing
live classifier logic
classifier-specific global state
```

The test performs only:

```text
Serial
  ↓
GetModel()
  ↓
Schema validation
  ↓
Create resolver
  ↓
Create MicroInterpreter
  ↓
AllocateTensors()
  ↓
Inspect tensors
  ↓
Fill dummy INT8 input
  ↓
Invoke()
  ↓
Inspect output
```

The test uses a deterministic input filled with the model input tensor's zero point.

This makes it possible to test model execution independently of the real IMU pipeline.

---

## Test 3 Compilation Status

Controlled Stripping Test 3 successfully compiles for the Arduino Nano 33 BLE.

Current compilation result:

```text
Sketch uses 181968 bytes (18%) of program storage space.
Maximum is 983040 bytes.

Global variables use 112104 bytes (42%) of dynamic memory,
leaving 150040 bytes for local variables.

Maximum is 262144 bytes.
```

Therefore, the diagnostic firmware itself fits comfortably within both flash and SRAM limits according to the Arduino build report.

At the time of this README update, the next step is to upload Test 3 and inspect its Serial Monitor output.

---

# Test 3 Decision Point

Test 3 is designed to distinguish between two fundamentally different failure categories.

## Case A — `AllocateTensors()` still fails

If execution again stops at:

```text
[A1] ENTER AllocateTensors()
```

without:

```text
[A2] EXIT AllocateTensors()
```

then the live classifier components are no longer plausible direct causes.

The investigation should then focus on the minimal combination of:

```text
Embedded TFLite model
+
TensorFlow Lite Micro implementation
+
MicroInterpreter
+
Operator implementations
+
Tensor arena / allocation behavior
+
Target-specific runtime behavior
```

Possible areas for subsequent investigation include library compatibility, model/operator compatibility, memory-planner behavior, and target-specific runtime issues.

These remain hypotheses until tested.

---

## Case B — `AllocateTensors()` succeeds

If the output reaches:

```text
[A1] ENTER AllocateTensors()
[A2] EXIT AllocateTensors()
Allocate status: 0
[A3] AllocateTensors SUCCESS
```

then the model and minimal TFLM environment can allocate successfully.

The investigation would then compare Test 2 against Test 3 and progressively reintroduce removed components until the specific trigger is found.

---

## Case C — Full minimal inference succeeds

The ideal result is:

```text
[A3] AllocateTensors SUCCESS

[T3] Input/output tensors OK

[F2] Input filled successfully

[V1] ENTER Invoke()
[V2] EXIT Invoke()

[V3] Invoke SUCCESS

*** MINIMAL TFLM TEST PASSED ***
```

This would demonstrate that the embedded model can be allocated and executed successfully in the minimal firmware.

The project could then move back toward reconstructing the real-time classifier.

---

# Debugging Strategy

The project deliberately uses a controlled stripping methodology instead of making unrelated changes simultaneously.

The process is:

```text
Full application fails
        ↓
Remove subsystem
        ↓
Repeat exact TFLM test
        ↓
Observe last successful checkpoint
        ↓
Remove another subsystem
        ↓
Repeat
        ↓
Identify minimal failure
```

Once a minimal passing configuration is found, the process can be reversed:

```text
Minimal working system
        ↓
Add one component
        ↓
Test
        ↓
Add next component
        ↓
Test
        ↓
Failure appears
        ↓
Identify triggering component
```

This approach prevents IMU acquisition, preprocessing, model inference, memory behavior, and serial/USB issues from being debugged simultaneously.

---

# Serial Diagnostics

The project currently uses explicit Serial checkpoints around potentially critical operations.

For example:

```cpp
checkpoint("[A1] ENTER AllocateTensors()");

TfLiteStatus alloc_status =
    interpreter.AllocateTensors();

checkpoint("[A2] EXIT AllocateTensors()");
```

If the board resets, crashes, or stops between the two messages, the failing operation can be localized.

The same strategy is used around:

```text
GetModel()
Resolver registration
MicroInterpreter construction
AllocateTensors()
Input/output tensor access
Invoke()
```

---

# USB / Serial Upload Issue

During development, an additional issue has occasionally occurred with the Arduino serial port.

Typical messages included:

```text
No device found on COM19
```

and:

```text
Serial port busy.
Could not connect to COM19 serial port.
```

This issue is separate from the TensorFlow Lite model investigation.

Successful firmware uploads have subsequently been observed, including complete flash programming such as:

```text
100%
Done in ~7–8 seconds
```

Therefore, USB/COM-port failures should not automatically be interpreted as TensorFlow Lite failures.

---

# Confirmed Findings So Far

The following points have been directly observed during the debugging process:

- The project compiles for Arduino Nano 33 BLE.
- The embedded model is approximately 10 KB.
- Model schema version and runtime schema version match.
- The model requires eight registered TensorFlow Lite operators.
- All eight operators can be added to the resolver.
- `MicroInterpreter` construction completes.
- Earlier stripped tests consistently reached `AllocateTensors()`.
- Test 2 still exhibited the issue after removing the IMU dependency.
- Removing `raw_scaler` did not eliminate the issue in Test 2.
- Removing the sample buffer did not eliminate the issue in Test 2.
- Removing live classifier globals did not eliminate the issue in Test 2.
- The 64 KB tensor arena is explicitly aligned.
- Controlled Stripping Test 3 compiles successfully.
- Test 3 uses approximately 42% of reported dynamic memory before runtime tensor allocation.

---

# Not Yet Proven

The following should **not yet be considered confirmed causes**:

```text
Tensor arena is too small
TFLM library is incompatible
The model is corrupted
A specific operator is broken
The IMU causes the crash
raw_scaler causes the crash
The CNN architecture itself is invalid
```

The controlled tests are specifically intended to distinguish between these possibilities.

---

# Current Project Status

As of the current development state:

```text
Model training/conversion        DONE
INT8 model export                DONE
Model embedded in firmware       DONE
Model schema validation          PASS
Operator identification          PASS
Resolver construction            PASS
Interpreter construction         PASS
Controlled stripping Test 1      DONE
Controlled stripping Test 2      DONE
Controlled stripping Test 3      COMPILED
Test 3 runtime execution         NEXT
Full live gesture classifier     BLOCKED pending TFLM diagnosis
```

The immediate next milestone is:

> Determine whether the absolute-minimal firmware can successfully execute `AllocateTensors()` and `Invoke()`.

---

# Expected Development Path

If Test 3 passes, development will proceed approximately as:

```text
Minimal TFLM inference
        ↓
Restore scaler
        ↓
Restore preprocessing
        ↓
Restore sample buffer
        ↓
Restore IMU
        ↓
Restore motion detection
        ↓
Restore gesture capture
        ↓
Run real gesture inference
        ↓
Validate predictions
        ↓
Optimize latency and memory
```

If Test 3 fails during tensor allocation, investigation will remain at the TensorFlow Lite Micro/model-runtime layer before any live gesture functionality is restored.

---

# Development Principle

A key principle of this project is:

> Do not optimize or rebuild the complete gesture pipeline until minimal model inference is proven stable on the target hardware.

This keeps the current debugging process reproducible and makes it possible to distinguish machine-learning/model problems from embedded-system integration problems.

---

## Current Milestone

**Milestone: Minimal on-device TFLite Micro inference**

Current checkpoint:

```text
Controlled Stripping Test 3
Arduino Nano 33 BLE
Depthwise CNN INT8
64 KB aligned tensor arena
Compilation: PASS
Runtime test: pending
```

Once this milestone passes, development can return to the complete real-time gesture-recognition pipeline.
