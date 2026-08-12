#include <Arduino.h>

#include <Arduino_BMI270_BMM150.h>
#include <Chirale_TensorFlowLite.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "standard_cnn_model.h"
#include "standard_cnn_scaler.h"

#include <math.h>


// ============================================================
// REAL-TIME STANDARD CNN GESTURE CLASSIFIER
// ------------------------------------------------------------
// Target:
// Arduino Nano 33 BLE
//
// Pipeline:
//
// IMU
//  ↓
// Motion detection
//  ↓
// 128 × 6 samples
//  ↓
// Standardization using raw_scaler.h
//  ↓
// INT8 quantization
//  ↓
// Standard CNN
//  ↓
// TensorFlow Lite Micro
//  ↓
// 4-class prediction
//
// IMPORTANT:
// standard_cnn_model.h must contain:
//
// alignas(16) const unsigned char standard_cnn_model[] = {...};
//
// Controlled Test 6 already proved:
// - direct Flash model works
// - alignment works
// - AllocateTensors() works
// - Invoke() works
// ============================================================


// ============================================================
// Model dimensions
// ============================================================

constexpr int kWindowSize = 128;
constexpr int kNumChannels = 6;
constexpr int kInputElements =
    kWindowSize * kNumChannels;

constexpr int kNumClasses = 4;


// ============================================================
// Class names
//
// Class order matches the training dataset and TFLite output.
// ============================================================

const char* kClassLabels[kNumClasses] =
{
    "circle",
    "left_right",
    "rest",
    "up_down"
};


// ============================================================
// Motion detection
//
// Uses deviation of acceleration magnitude from 1 g.
//
// Example:
//
// stationary:
// sqrt(ax² + ay² + az²) ≈ 1
//
// gesture:
// deviation increases
// ============================================================

constexpr float kMotionThreshold = 0.20f;


// ============================================================
// Cooldown
//
// Prevent immediate repeated triggering after one gesture.
// ============================================================

constexpr unsigned long kCooldownMs = 700;


// ============================================================
// Tensor Arena
//
// Test 6 observed:
//
// Arena used ≈ 6852 bytes
//
// We keep 64 KB because this configuration is already proven.
// ============================================================

constexpr size_t kTensorArenaSize =
    64 * 1024;

alignas(16)
static uint8_t tensorArena[
    kTensorArenaSize
];


// ============================================================
// Raw IMU sample buffer
//
// Format:
//
// row:
// ax ay az gx gy gz
//
// 128 × 6 floats
// ============================================================

static float sampleBuffer[
    kWindowSize
][
    kNumChannels
];


// ============================================================
// TensorFlow Lite objects
// ============================================================

const tflite::Model* gModel =
    nullptr;

TfLiteTensor* gInput =
    nullptr;

TfLiteTensor* gOutput =
    nullptr;


// ============================================================
// Resolver
// ============================================================

static
tflite::MicroMutableOpResolver<7>
    gResolver;


// ============================================================
// Interpreter pointer
// ============================================================

tflite::MicroInterpreter*
    gInterpreter =
    nullptr;


// ============================================================
// Runtime state
// ============================================================

unsigned long lastGestureTime =
    0;


// ============================================================
// Utility
// ============================================================

void fatalStop(
    const char* message
)
{
    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "FATAL ERROR"
    );

    Serial.println(
        "========================================"
    );

    Serial.println(
        message
    );

    Serial.flush();

    while (true)
    {
        delay(1000);
    }
}


// ============================================================
// Clamp INT8
// ============================================================

int8_t clampInt8(
    int32_t value
)
{
    if (value > 127)
    {
        value = 127;
    }

    if (value < -128)
    {
        value = -128;
    }

    return
        static_cast<int8_t>(
            value
        );
}


// ============================================================
// Read one synchronized 6-axis IMU sample
//
// Output order:
//
// 0 ax
// 1 ay
// 2 az
// 3 gx
// 4 gy
// 5 gz
// ============================================================

bool readIMUSample(
    float& ax,
    float& ay,
    float& az,
    float& gx,
    float& gy,
    float& gz
)
{
    if (
        !IMU.accelerationAvailable()
        ||
        !IMU.gyroscopeAvailable()
    )
    {
        return false;
    }


    IMU.readAcceleration(
        ax,
        ay,
        az
    );


    IMU.readGyroscope(
        gx,
        gy,
        gz
    );


    return true;
}


// ============================================================
// Motion score
//
// accelerationMagnitude ≈ 1 g at rest.
//
// We use:
//
// abs(|a| - 1)
//
// as a simple trigger.
// ============================================================

float calculateMotionScore(
    float ax,
    float ay,
    float az
)
{
    const float magnitude =
        sqrtf(
            ax * ax
            +
            ay * ay
            +
            az * az
        );


    return
        fabsf(
            magnitude
            -
            1.0f
        );
}


// ============================================================
// Store one sample
// ============================================================

void storeSample(
    int index,
    float ax,
    float ay,
    float az,
    float gx,
    float gy,
    float gz
)
{
    sampleBuffer[index][0] =
        ax;

    sampleBuffer[index][1] =
        ay;

    sampleBuffer[index][2] =
        az;

    sampleBuffer[index][3] =
        gx;

    sampleBuffer[index][4] =
        gy;

    sampleBuffer[index][5] =
        gz;
}


// ============================================================
// Capture complete gesture window
//
// Trigger sample is stored as sample 0.
// Then capture continues until 128 samples are available.
// ============================================================

bool captureGestureWindow(
    float firstAx,
    float firstAy,
    float firstAz,
    float firstGx,
    float firstGy,
    float firstGz
)
{
    Serial.println();
    Serial.println(
        "Motion detected."
    );

    Serial.println(
        "Capturing 128 x 6 IMU window..."
    );


    storeSample(
        0,
        firstAx,
        firstAy,
        firstAz,
        firstGx,
        firstGy,
        firstGz
    );


    int sampleIndex =
        1;


    unsigned long timeoutStart =
        millis();


    while (
        sampleIndex
        <
        kWindowSize
    )
    {
        float ax;
        float ay;
        float az;

        float gx;
        float gy;
        float gz;


        if (
            readIMUSample(
                ax,
                ay,
                az,
                gx,
                gy,
                gz
            )
        )
        {
            storeSample(
                sampleIndex,
                ax,
                ay,
                az,
                gx,
                gy,
                gz
            );


            sampleIndex++;

            timeoutStart =
                millis();
        }


        // ----------------------------------------------------
        // Safety timeout
        // ----------------------------------------------------

        if (
            millis()
            -
            timeoutStart
            >
            2000
        )
        {
            Serial.println(
                "ERROR: IMU capture timeout."
            );

            return false;
        }
    }


    Serial.println(
        "Capture complete."
    );


    return true;
}


// ============================================================
// Preprocess and quantize
//
// Training preprocessing:
//
// standardized =
//     (raw - mean) / std
//
// Then:
//
// q = standardized / input_scale
//     + input_zero_point
//
// Finally clamp to INT8.
// ============================================================

bool preprocessAndQuantize()
{
    if (
        gInput == nullptr
    )
    {
        return false;
    }


    if (
        gInput->type
        !=
        kTfLiteInt8
    )
    {
        Serial.println(
            "ERROR: model input is not INT8."
        );

        return false;
    }


    const float inputScale =
        gInput
            ->params
            .scale;


    const int inputZeroPoint =
        gInput
            ->params
            .zero_point;


    if (
        inputScale
        <=
        0.0f
    )
    {
        Serial.println(
            "ERROR: invalid input scale."
        );

        return false;
    }


    int tensorIndex =
        0;


    for (
        int sample = 0;
        sample < kWindowSize;
        sample++
    )
    {
        for (
            int channel = 0;
            channel < kNumChannels;
            channel++
        )
        {
            const float rawValue =
                sampleBuffer
                    [sample]
                    [channel];


            const float stdValue =
                raw_std[
                    channel
                ];


            if (
                fabsf(stdValue)
                <
                1e-9f
            )
            {
                Serial.println(
                    "ERROR: raw_std contains zero."
                );

                return false;
            }


            // -----------------------------------------------
            // Standardization
            // -----------------------------------------------

            const float standardized =
                (
                    rawValue
                    -
                    raw_mean[
                        channel
                    ]
                )
                /
                stdValue;


            // -----------------------------------------------
            // INT8 quantization
            // -----------------------------------------------

            const float quantizedFloat =
                standardized
                /
                inputScale
                +
                inputZeroPoint;


            const int32_t quantized =
                static_cast<int32_t>(
                    roundf(
                        quantizedFloat
                    )
                );


            gInput
                ->data
                .int8[
                    tensorIndex
                ]
                =
                clampInt8(
                    quantized
                );


            tensorIndex++;
        }
    }


    return
        tensorIndex
        ==
        kInputElements;
}


// ============================================================
// Print prediction
// ============================================================

void printPrediction()
{
    if (
        gOutput == nullptr
    )
    {
        return;
    }


    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "GESTURE PREDICTION"
    );

    Serial.println(
        "========================================"
    );


    int bestClass =
        0;


    float bestScore =
        -1.0f;


    for (
        int i = 0;
        i < kNumClasses;
        i++
    )
    {
        const int8_t raw =
            gOutput
                ->data
                .int8[i];


        const float score =
            (
                static_cast<int>(
                    raw
                )
                -
                gOutput
                    ->params
                    .zero_point
            )
            *
            gOutput
                ->params
                .scale;


        Serial.print(
            kClassLabels[i]
        );

        Serial.print(
            ": "
        );

        Serial.println(
            score,
            6
        );


        if (
            score
            >
            bestScore
        )
        {
            bestScore =
                score;

            bestClass =
                i;
        }
    }


    Serial.println();


    Serial.print(
        "Predicted class: "
    );

    Serial.println(
        bestClass
    );


    Serial.print(
        "Gesture: "
    );

    Serial.println(
        kClassLabels[
            bestClass
        ]
    );


    Serial.print(
        "Confidence: "
    );

    Serial.println(
        bestScore,
        6
    );


    Serial.println(
        "========================================"
    );
}


// ============================================================
// Run inference
// ============================================================

bool runInference()
{
    const uint32_t preprocessingStart =
        micros();


    if (
        !preprocessAndQuantize()
    )
    {
        Serial.println(
            "Preprocessing failed."
        );

        return false;
    }


    const uint32_t preprocessingEnd =
        micros();


    const uint32_t inferenceStart =
        micros();


    const TfLiteStatus status =
        gInterpreter
            ->Invoke();


    const uint32_t inferenceEnd =
        micros();


    if (
        status
        !=
        kTfLiteOk
    )
    {
        Serial.println(
            "Invoke() failed."
        );

        return false;
    }


    const uint32_t preprocessingUs =
        preprocessingEnd
        -
        preprocessingStart;


    const uint32_t inferenceUs =
        inferenceEnd
        -
        inferenceStart;


    Serial.println();

    Serial.print(
        "Preprocessing: "
    );

    Serial.print(
        preprocessingUs
    );

    Serial.println(
        " us"
    );


    Serial.print(
        "Inference: "
    );

    Serial.print(
        inferenceUs
    );

    Serial.println(
        " us"
    );


    Serial.print(
        "Inference: "
    );

    Serial.print(
        inferenceUs
        /
        1000.0f,
        3
    );

    Serial.println(
        " ms"
    );


    printPrediction();


    return true;
}


// ============================================================
// TensorFlow initialization
// ============================================================

void initializeTensorFlow()
{
    Serial.println(
        "[TFLM] Checking model alignment..."
    );


    const uintptr_t modelAddress =
        reinterpret_cast<uintptr_t>(
            standard_cnn_model
        );


    Serial.print(
        "[TFLM] Model address: 0x"
    );

    Serial.println(
        modelAddress,
        HEX
    );


    Serial.print(
        "[TFLM] Model mod16: "
    );

    Serial.println(
        modelAddress % 16
    );


    if (
        modelAddress % 16
        !=
        0
    )
    {
        fatalStop(
            "standard_cnn_model is not 16-byte aligned"
        );
    }


    // ========================================================
    // GetModel
    // ========================================================

    gModel =
        tflite::GetModel(
            standard_cnn_model
        );


    if (
        gModel == nullptr
    )
    {
        fatalStop(
            "GetModel() failed"
        );
    }


    if (
        gModel->version()
        !=
        TFLITE_SCHEMA_VERSION
    )
    {
        fatalStop(
            "TFLite schema mismatch"
        );
    }


    // ========================================================
    // Resolver
    // ========================================================

    if (
        gResolver.AddExpandDims()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddExpandDims failed"
        );
    }


    if (
        gResolver.AddConv2D()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddConv2D failed"
        );
    }


    if (
        gResolver.AddReshape()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddReshape failed"
        );
    }


    if (
        gResolver.AddMaxPool2D()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddMaxPool2D failed"
        );
    }


    if (
        gResolver.AddMean()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddMean failed"
        );
    }


    if (
        gResolver.AddFullyConnected()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddFullyConnected failed"
        );
    }


    if (
        gResolver.AddSoftmax()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AddSoftmax failed"
        );
    }


    // ========================================================
    // Interpreter
    // ========================================================

    static
    tflite::MicroInterpreter
        staticInterpreter(
            gModel,
            gResolver,
            tensorArena,
            kTensorArenaSize
        );


    gInterpreter =
        &staticInterpreter;


    // ========================================================
    // Allocate tensors
    // ========================================================

    Serial.println(
        "[TFLM] AllocateTensors()..."
    );


    if (
        gInterpreter
            ->AllocateTensors()
        !=
        kTfLiteOk
    )
    {
        fatalStop(
            "AllocateTensors() failed"
        );
    }


    Serial.println(
        "[TFLM] AllocateTensors() PASS"
    );


    Serial.print(
        "[TFLM] Arena used: "
    );

    Serial.println(
        gInterpreter
            ->arena_used_bytes()
    );


    // ========================================================
    // Input / output
    // ========================================================

    gInput =
        gInterpreter
            ->input(0);


    gOutput =
        gInterpreter
            ->output(0);


    if (
        gInput == nullptr
        ||
        gOutput == nullptr
    )
    {
        fatalStop(
            "Input/output tensor unavailable"
        );
    }


    if (
        gInput->type
        !=
        kTfLiteInt8
    )
    {
        fatalStop(
            "Expected INT8 input"
        );
    }


    if (
        gOutput->type
        !=
        kTfLiteInt8
    )
    {
        fatalStop(
            "Expected INT8 output"
        );
    }


    if (
        gInput->bytes
        !=
        kInputElements
    )
    {
        fatalStop(
            "Unexpected model input size"
        );
    }


    if (
        gOutput->bytes
        !=
        kNumClasses
    )
    {
        fatalStop(
            "Unexpected model output size"
        );
    }


    Serial.print(
        "[TFLM] Input scale: "
    );

    Serial.println(
        gInput
            ->params
            .scale,
        8
    );


    Serial.print(
        "[TFLM] Input zero point: "
    );

    Serial.println(
        gInput
            ->params
            .zero_point
    );


    Serial.print(
        "[TFLM] Output scale: "
    );

    Serial.println(
        gOutput
            ->params
            .scale,
        8
    );


    Serial.print(
        "[TFLM] Output zero point: "
    );

    Serial.println(
        gOutput
            ->params
            .zero_point
    );


    Serial.println(
        "[TFLM] READY"
    );
}


// ============================================================
// IMU initialization
// ============================================================

void initializeIMU()
{
    Serial.println(
        "[IMU] Initializing..."
    );


    if (
        !IMU.begin()
    )
    {
        fatalStop(
            "IMU.begin() failed"
        );
    }


    Serial.println(
        "[IMU] READY"
    );


    Serial.print(
        "[IMU] Accelerometer sample rate: "
    );

    Serial.print(
        IMU.accelerationSampleRate()
    );

    Serial.println(
        " Hz"
    );


    Serial.print(
        "[IMU] Gyroscope sample rate: "
    );

    Serial.print(
        IMU.gyroscopeSampleRate()
    );

    Serial.println(
        " Hz"
    );
}


// ============================================================
// Setup
// ============================================================

void setup()
{
    Serial.begin(
        115200
    );


    const unsigned long serialStart =
        millis();


    while (
        !Serial
        &&
        (
            millis()
            -
            serialStart
            <
            3000
        )
    )
    {
        delay(10);
    }


    delay(
        500
    );


    Serial.println();

    Serial.println(
        "========================================"
    );

    Serial.println(
        "STANDARD CNN INT8"
    );

    Serial.println(
        "REAL-TIME GESTURE CLASSIFIER"
    );

    Serial.println(
        "Arduino Nano 33 BLE"
    );

    Serial.println(
        "========================================"
    );


    // ========================================================
    // TensorFlow first
    // ========================================================

    initializeTensorFlow();


    // ========================================================
    // IMU
    // ========================================================

    initializeIMU();


    Serial.println();

    Serial.println(
        "========================================"
    );

    Serial.println(
        "SYSTEM READY"
    );

    Serial.println(
        "Move the board to perform a gesture."
    );

    Serial.print(
        "Motion threshold: "
    );

    Serial.println(
        kMotionThreshold,
        4
    );

    Serial.println(
        "========================================"
    );

    Serial.println();
}


// ============================================================
// Main loop
// ============================================================

void loop()
{
    float ax;
    float ay;
    float az;

    float gx;
    float gy;
    float gz;


    // ========================================================
    // Wait for synchronized IMU data
    // ========================================================

    if (
        !readIMUSample(
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        )
    )
    {
        return;
    }


    // ========================================================
    // Cooldown
    // ========================================================

    if (
        millis()
        -
        lastGestureTime
        <
        kCooldownMs
    )
    {
        return;
    }


    // ========================================================
    // Detect motion
    // ========================================================

    const float motionScore =
        calculateMotionScore(
            ax,
            ay,
            az
        );


    if (
        motionScore
        <
        kMotionThreshold
    )
    {
        return;
    }


    Serial.print(
        "Motion score: "
    );

    Serial.println(
        motionScore,
        4
    );


    // ========================================================
    // Capture
    // ========================================================

    if (
        !captureGestureWindow(
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        )
    )
    {
        return;
    }


    // ========================================================
    // Run CNN
    // ========================================================

    if (
        runInference()
    )
    {
        lastGestureTime =
            millis();
    }


    Serial.println();

    Serial.println(
        "Ready for next gesture."
    );

    Serial.println();
}