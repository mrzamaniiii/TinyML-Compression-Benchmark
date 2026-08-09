TinyML Depthwise CNN Gesture Classifier on Arduino Nano 33 BLE

An embedded TinyML project for real-time gesture classification on the Arduino Nano 33 BLE using IMU data and a quantized INT8 Depthwise Convolutional Neural Network (CNN) deployed with TensorFlow Lite Micro.

The project is currently in the deployment and low-level debugging phase. The trained model has been successfully converted to a TensorFlow Lite model and embedded into the Arduino firmware. The current investigation focuses on the TensorFlow Lite Micro tensor allocation stage on the target microcontroller.

Project Overview

The goal of this project is to deploy a lightweight neural network for gesture recognition directly on a microcontroller.

The intended inference pipeline is:

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

The final system is intended to perform the complete classification pipeline locally on the Arduino Nano 33 BLE without requiring cloud inference or a connected computer.

Hardware

Target board:

Arduino Nano 33 BLE

ARM Cortex-M4F

On-board IMU

256 KB SRAM

1 MB Flash

Development is currently performed through the Arduino IDE using USB serial communication for diagnostics.

Software Environment

Current development environment:

Arduino IDE: 2.3.8
Target: Arduino Nano 33 BLE
Serial baud rate: 115200

TensorFlow Lite Micro is integrated through:

#include <Chirale_TensorFlowLite.h>

with the required TensorFlow Lite Micro headers.

Neural Network

The deployed model is a quantized:

Depthwise CNN
INT8

Current embedded model size:

10,368 bytes

The model uses TensorFlow Lite schema version:

Model schema version:   3
Runtime schema version: 3

Therefore, no model/runtime schema mismatch has been observed.

TensorFlow Lite Operators

Inspection of the embedded model showed that it requires eight TensorFlow Lite operators.

The current resolver contains:

tflite::MicroMutableOpResolver<8>

with:

ExpandDims
DepthwiseConv2D
Conv2D
Reshape
MaxPool2D
Mean
FullyConnected
Softmax

The corresponding model operator codes observed during diagnostics were:

Opcode 0: builtin=70, version=1
Opcode 1: builtin=4,  version=3
Opcode 2: builtin=3,  version=3
Opcode 3: builtin=22, version=1
Opcode 4: builtin=17, version=2
Opcode 5: builtin=40, version=2
Opcode 6: builtin=9,  version=4
Opcode 7: builtin=25, version=2

All required operators can be successfully registered in the MicroMutableOpResolver.

Tensor Arena

The current diagnostic configuration uses a 64 KB tensor arena:

constexpr int TENSOR_ARENA_SIZE = 64 * 1024;

alignas(16)
static uint8_t tensorArena[TENSOR_ARENA_SIZE];

Arena capacity:

65,536 bytes

Explicit alignment is used to eliminate obvious tensor-arena alignment problems during testing.

Original Runtime Architecture

The intended full firmware contains several major components.

1. IMU acquisition

The Arduino's IMU provides motion data used to capture gestures.

2. Motion detection

A configurable motion threshold determines when a gesture capture should begin.

A diagnostic configuration used:

Motion threshold: 0.2000

3. Sample buffer

Captured IMU samples are stored before preprocessing.

One observed configuration reported:

Sample buffer bytes: 3072

4. Raw scaler

Training-time scaling parameters are embedded into the firmware through:

raw_scaler.h

These parameters are intended to reproduce the preprocessing used during model training.

5. INT8 quantization

Preprocessed samples are converted into the quantized representation expected by the TensorFlow Lite model.

6. TensorFlow Lite Micro inference

The intended inference sequence is:

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

Project Files

The current Arduino project contains the following important files:

Depthwise_Gesture_Classifier.ino
raw_scaler.h
depthwise_model.h

Depthwise_Gesture_Classifier.ino

Main firmware and diagnostic program.

Depending on the debugging stage, this file contains either the complete live gesture-classification pipeline or a stripped TensorFlow Lite Micro diagnostic test.

raw_scaler.h

Contains preprocessing/scaling parameters used by the original classifier.

This dependency has also been deliberately removed in controlled tests to determine whether it contributes to the current runtime failure.

depthwise_model.h

Contains the embedded TensorFlow Lite INT8 model as a C/C++ byte array.

Current model size:

10,368 bytes

Current Debugging Investigation

During integration of the complete live classifier, execution repeatedly stopped during:

interpreter.AllocateTensors();

To locate the failure precisely, extensive serial checkpoints were added.

For example:

[D1] Loading model
[D2] GetModel returned
[D3] Schema OK

[R0] Creating resolver
...
[R9] Resolver ready

[I1] Creating interpreter
[I2] Interpreter created

[A1] ENTER AllocateTensors()

The critical observation was that execution reached:

[A1] ENTER AllocateTensors()

but did not reach the checkpoint immediately after AllocateTensors().

This strongly localizes the current failure to execution occurring during tensor allocation.

Controlled Stripping Tests

Rather than changing multiple components simultaneously, the firmware has been progressively stripped down to isolate the failing subsystem.

Controlled Stripping Test 1

The first stripped configuration removed major parts of the live classification pipeline while retaining enough of the original application structure to test TensorFlow Lite initialization.

The model was successfully:

located in memory,

parsed,

schema-validated,

registered with the required operators,

passed to the interpreter.

Execution nevertheless stopped during:

AllocateTensors()

This suggested that the failure was not simply occurring during model loading or resolver construction.

Controlled Stripping Test 2

The second test removed additional classifier-specific dependencies.

The Serial Monitor explicitly confirmed:

CONTROLLED STRIPPING TEST 2

NO IMU dependency
NO raw_scaler
NO sampleBuffer
NO live classifier globals

The initialization sequence continued successfully through:

[D1] Loading model
[D2] GetModel returned
[D3] Schema OK

followed by successful operator registration:

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

and successful interpreter construction:

[I1] Creating interpreter
[I2] Interpreter created

The final observed checkpoint was again:

[A1] ENTER AllocateTensors()

Therefore, removing the following components did not eliminate the problem:

IMU dependency
raw_scaler
sampleBuffer
live classifier globals

This significantly narrowed the investigation.

Controlled Stripping Test 3

Controlled Stripping Test 3 reduced the firmware to an absolute-minimal TensorFlow Lite Micro runtime.

The firmware contained no:

IMU
raw_scaler
sampleBuffer
motion detection
gesture capture
preprocessing
live classifier logic
classifier-specific global state

The test performed only:

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
Fill deterministic INT8 input
  ↓
Invoke()
  ↓
Inspect output

The input tensor was filled with its quantized zero point so that inference could be tested independently of the live IMU pipeline.

Test 3 Result

Test 3 passed completely on the Arduino Nano 33 BLE.

Observed tensor allocation result:

[A1] ENTER AllocateTensors()
[A2] EXIT AllocateTensors()
Allocate status: 0
[A3] AllocateTensors SUCCESS

Observed input tensor:

Type: 9
Bytes: 768
Scale: 0.04553470
Zero point: 13
Dimensions: 1 x 128 x 6

Observed output tensor:

Type: 9
Bytes: 4
Scale: 0.00390625
Zero point: -128
Dimensions: 1 x 4

Inference also completed successfully:

[V1] ENTER Invoke()
[V2] EXIT Invoke()
Invoke status: 0
[V3] Invoke SUCCESS

The deterministic test input produced:

Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625

The test concluded with:

*** MINIMAL TFLM TEST PASSED ***

This proved that the following components were functional together on the target board:

depthwise_model.h
TensorFlow Lite schema
MicroMutableOpResolver
MicroInterpreter
64 KB tensor arena
AllocateTensors()
INT8 input/output tensors
Invoke()

This was the first decisive indication that the Depthwise CNN and TensorFlow Lite Micro runtime were not inherently broken.

Controlled Stripping Test 4

Test 4 started from the passing Test 3 baseline and added only:

raw_scaler.h

No IMU, sample buffer, preprocessing pipeline, or live classifier logic was restored.

The scaler arrays were deliberately referenced at runtime so that they were definitely linked into the firmware.

The scaler values were read successfully:

[S1] ENTER raw_scaler test
...
[S2] raw_scaler reference SUCCESS

The model was still parsed successfully, the schema matched, all eight operators were registered, and the interpreter constructor completed.

However, execution again stopped at:

[A1] ENTER AllocateTensors()

with no corresponding:

[A2] EXIT AllocateTensors()

Test 4 Interpretation

This result initially suggested that the presence of raw_scaler.h changed something important in the executable.

However, the Arduino memory report did not show a meaningful increase in static RAM usage compared with Test 3.

The scaler arrays also appeared at low memory addresses associated with program/constant storage rather than the main SRAM region.

Therefore, the leading hypothesis shifted from "the scaler consumes too much RAM" to:

Adding raw_scaler.h changes the binary layout and therefore changes the placement/alignment of the embedded TFLite model.

This hypothesis was tested directly in Test 5.

Controlled Stripping Test 5

Test 5 kept raw_scaler.h present exactly as in Test 4, but copied the entire TFLite model into an explicitly 16-byte-aligned RAM buffer before calling GetModel().

The test therefore compared:

Original embedded model in program memory
vs.
16-byte-aligned RAM copy of the exact same model bytes

The model copy was verified byte-for-byte before execution.

Model Alignment Observation

The original embedded model address was:

Original model addr:  0x330A4
Original model mod16: 4

Therefore, the original FlatBuffer was not 16-byte aligned.

The RAM copy was:

Aligned model addr:   0x20011030
Aligned model mod16:  0

The tensor arena was also 16-byte aligned.

Test 5 Result

With raw_scaler.h still present and the model loaded from the aligned RAM copy, tensor allocation succeeded:

[A1] ENTER AllocateTensors()
[A2] EXIT AllocateTensors()
Allocate status: 0
[A3] AllocateTensors SUCCESS

The actual arena usage was:

Arena used: 6852

out of:

Arena capacity: 65536

This strongly demonstrates that the failure was not caused by an undersized tensor arena.

Input/output tensors were valid:

Input bytes: 768
Input scale: 0.04553470
Input zero point: 13

Output bytes: 4
Output scale: 0.00390625
Output zero point: -128

Inference also succeeded:

[V1] ENTER Invoke()
[V2] EXIT Invoke()
Invoke status: 0
[V3] Invoke SUCCESS

Observed output:

Class 0: 0.07812500
Class 1: 0.28125000
Class 2: 0.51171875
Class 3: 0.12890625

The test concluded with:

CONTROLLED STRIPPING TEST 5 PASSED

raw_scaler.h: PRESENT
Model source: 16-byte aligned RAM copy
AllocateTensors(): PASS
Invoke(): PASS

Current Root-Cause Finding

The current evidence strongly indicates that the original failure was associated with the placement/alignment of the embedded TFLite model FlatBuffer.

The key comparison is:

Test 3:
minimal firmware
model placement happened to work
AllocateTensors(): PASS
Invoke(): PASS

Test 4:
raw_scaler.h added
embedded model placement changed
AllocateTensors(): FAIL

Test 5:
raw_scaler.h still present
same model bytes copied to alignas(16) RAM
AllocateTensors(): PASS
Invoke(): PASS

The original model address in Test 5 had:

address mod 16 = 4

while the working RAM copy had:

address mod 16 = 0

This makes model alignment/placement the strongest current explanation for the inconsistent behavior.

This is still best described as a strongly supported root-cause finding, rather than a final production fix, because the current working workaround duplicates the model in RAM.

Current Workaround

The currently proven workaround is:

alignas(16)
static uint8_t alignedModel[depthwise_model_len];

memcpy(
    alignedModel,
    depthwise_model,
    depthwise_model_len
);

const tflite::Model* model =
    tflite::GetModel(alignedModel);

This successfully allows both:

AllocateTensors()
Invoke()

to execute with raw_scaler.h present.

The disadvantage is that the full model is copied into SRAM, consuming approximately another 10 KB of RAM.

Therefore, this is considered a diagnostic/workaround solution rather than the desired final implementation.

Preferred Production Fix

The next technical objective is to keep the TFLite model in program memory while guaranteeing correct alignment at declaration time.

A future version of the generated model header should use an explicit alignment attribute such as:

alignas(16)
const unsigned char depthwise_model[] = {
    ...
};

or another compiler-supported alignment mechanism appropriate for the Arduino Nano 33 BLE toolchain.

The exact final declaration should be validated on the target board by printing:

(uintptr_t)depthwise_model % 16

and confirming that it equals:

0

before removing the RAM-copy workaround.

Debugging Strategy

The project deliberately used a controlled stripping methodology instead of making unrelated changes simultaneously.

The process was:

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
Find minimal passing configuration
        ↓
Reintroduce one dependency
        ↓
Failure reappears
        ↓
Form targeted hypothesis
        ↓
Run isolated confirmation test

This approach prevented IMU acquisition, preprocessing, TensorFlow Lite inference, memory behavior, and serial/USB issues from being debugged simultaneously.

Serial Diagnostics

Explicit serial checkpoints were used around critical runtime operations.

For example:

checkpoint("[A1] ENTER AllocateTensors()");

TfLiteStatus alloc_status =
    interpreter.AllocateTensors();

checkpoint("[A2] EXIT AllocateTensors()");

The same pattern was used around:

GetModel()
Resolver registration
MicroInterpreter construction
AllocateTensors()
Input/output tensor access
Invoke()

This made it possible to distinguish a returned TensorFlow Lite error from a crash/hang occurring inside the called function.

USB / Serial Upload Issue

During development, a separate Arduino USB/serial issue occasionally appeared:

No device found on COM19

and:

Serial port busy.
Could not connect to COM19 serial port.

This issue is separate from the TensorFlow Lite runtime problem.

Successful uploads were repeatedly confirmed by output such as:

100%
Done in ~7–8 seconds

When the port disappeared, entering the board bootloader with a double reset and reselecting the active COM port restored upload functionality.

Confirmed Findings So Far

The following points have now been directly observed:

The project compiles for Arduino Nano 33 BLE.

The embedded INT8 Depthwise CNN is 10,368 bytes.

Model schema version and runtime schema version both equal 3.

The model requires eight TensorFlow Lite operators.

All eight operators can be registered successfully.

MicroInterpreter construction succeeds.

A 64 KB aligned tensor arena is sufficient for this model.

Actual observed tensor-arena usage is approximately 6,852 bytes.

Test 3 proved that minimal AllocateTensors() and Invoke() succeed.

Test 4 showed that adding raw_scaler.h can change execution behavior even though scaler values themselves are valid.

raw_scaler.h is not inherently incompatible with the model.

Test 5 proved that raw_scaler.h and TFLM inference work together when the model is loaded from a 16-byte-aligned RAM copy.

The original model placement observed in Test 5 was not 16-byte aligned (mod16 = 4).

The working RAM copy was 16-byte aligned (mod16 = 0).

The model itself is not corrupted.

The TensorFlow Lite Micro runtime is able to execute the model successfully on the Arduino Nano 33 BLE.

IMU code is not required to reproduce the original allocation failure.

The failure is not explained by insufficient tensor-arena capacity.

Current Leading Conclusion

The strongest current conclusion is:

The embedded TFLite FlatBuffer must be placed with suitable alignment. Changes elsewhere in the firmware can alter the program-memory layout and expose an alignment-sensitive failure during AllocateTensors().

The proven temporary solution is to execute the model from a 16-byte-aligned RAM copy.

The preferred final solution is to enforce suitable alignment on the model array itself so the model can remain in program memory without wasting SRAM.

Current Project Status

Model training/conversion             DONE
INT8 model export                     DONE
Model embedded in firmware            DONE
Model schema validation               PASS
Operator identification               PASS
Resolver construction                 PASS
Interpreter construction              PASS

Controlled Stripping Test 1           DONE
Controlled Stripping Test 2           DONE
Controlled Stripping Test 3           PASS
Controlled Stripping Test 4           FAIL at AllocateTensors()
Controlled Stripping Test 5           PASS

Minimal on-device TFLM inference      PROVEN
raw_scaler + TFLM inference           PROVEN with aligned RAM model
Tensor arena sizing                   SUFFICIENT
Model alignment issue                 STRONGLY SUPPORTED

Final aligned Flash model declaration NEXT
Full live gesture classifier          NEXT AFTER ALIGNMENT FIX

Immediate Next Steps

The current recommended development sequence is:

1. Regenerate or edit depthwise_model.h
   so depthwise_model is explicitly 16-byte aligned
        ↓
2. Confirm model address mod16 == 0 on the board
        ↓
3. Run AllocateTensors() directly from Flash/program memory
   without the RAM model copy
        ↓
4. Confirm Invoke() succeeds
        ↓
5. Restore raw scaling/preprocessing
        ↓
6. Restore sample buffer
        ↓
7. Restore IMU initialization and live capture
        ↓
8. Run real gesture inference
        ↓
9. Validate prediction quality
        ↓
10. Benchmark latency, RAM, Flash, and accuracy

Expected Final Runtime Architecture

The intended production pipeline remains:

Arduino Nano 33 BLE IMU
        ↓
Motion detection
        ↓
128 × 6 sample window
        ↓
raw_mean / raw_std scaling
        ↓
INT8 quantization
        ↓
Depthwise CNN
(aligned embedded model)
        ↓
TensorFlow Lite Micro
        ↓
4-class probability output
        ↓
argmax
        ↓
Gesture label + confidence

The RAM model copy used in Test 5 is not intended to remain in the final optimized firmware if model alignment can be guaranteed directly in depthwise_model.h.

Development Principle

A key principle of this project remains:

Prove each embedded subsystem independently before rebuilding the complete real-time classifier.

The project has now successfully passed the minimal TensorFlow Lite Micro milestone and has isolated a strongly supported model-placement/alignment issue.

The next milestone is to convert the Test 5 workaround into a clean production alignment fix and then restore the live IMU gesture-classification pipeline.

Current Milestone

Milestone: Minimal on-device TFLite Micro inference — PASSED

Current checkpoint:

Arduino Nano 33 BLE
Depthwise CNN INT8
Model size: 10,368 bytes
Tensor arena: 64 KB
Observed arena use: 6,852 bytes
AllocateTensors(): PASS
Invoke(): PASS
raw_scaler.h: compatible
Aligned RAM model workaround: PASS

Next milestone:

16-byte-aligned model in program memory
        ↓
Full live IMU gesture classifier
