#include <Arduino.h>

#include <Arduino_BMI270_BMM150.h>
#include <Chirale_TensorFlowLite.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "rms_tiny_model.h"
#include "rms_feature_scaler.h"

#include <math.h>

// ============================================================
// REAL-TIME RMS TINY INT8 GESTURE CLASSIFIER
// ------------------------------------------------------------
// Target:
// Arduino Nano 33 BLE / BLE Sense Rev2
//
// Pipeline:
//
// IMU
//  ↓
// Motion detection
//  ↓
// 128 × 6 samples
//  ↓
// RMS extraction (1 feature per channel)
//  ↓
// 6 RMS features
//  ↓
// Feature standardization
//  ↓
// INT8 quantization
//  ↓
// Tiny Dense classifier (6 → 16 → 8 → 4)
//  ↓
// 4-class prediction
// ============================================================


// ============================================================
// Dimensions
// ============================================================

constexpr int kWindowSize = 128;
constexpr int kNumChannels = 6;
constexpr int kNumFeatures = 6;
constexpr int kNumClasses = 4;


// ============================================================
// Class labels
// ============================================================

const char* kClassLabels[kNumClasses] =
{
    "circle",
    "left_right",
    "rest",
    "up_down"
};


// ============================================================
// Motion detection / cooldown
// ============================================================

constexpr float kMotionThreshold = 0.20f;
constexpr unsigned long kCooldownMs = 700;


// ============================================================
// Tensor Arena
//
// Keep 64 KB initially so this benchmark uses the same
// allocation strategy as the CNN benchmarks.
// Actual usage is printed with arena_used_bytes().
// ============================================================

constexpr size_t kTensorArenaSize = 64 * 1024;

alignas(16)
static uint8_t tensorArena[kTensorArenaSize];


// ============================================================
// Buffers
// ============================================================

static float sampleBuffer[kWindowSize][kNumChannels];
static float rmsFeatures[kNumFeatures];


// ============================================================
// TFLite objects
// ============================================================

const tflite::Model* gModel = nullptr;
TfLiteTensor* gInput = nullptr;
TfLiteTensor* gOutput = nullptr;

static tflite::MicroMutableOpResolver<2> gResolver;

tflite::MicroInterpreter* gInterpreter = nullptr;


// ============================================================
// Runtime state
// ============================================================

unsigned long lastGestureTime = 0;


// ============================================================
// Utility
// ============================================================

void fatalStop(const char* message)
{
    Serial.println();
    Serial.println("========================================");
    Serial.println("FATAL ERROR");
    Serial.println("========================================");
    Serial.println(message);
    Serial.flush();

    while (true)
    {
        delay(1000);
    }
}


int8_t clampInt8(int32_t value)
{
    if (value > 127) value = 127;
    if (value < -128) value = -128;

    return static_cast<int8_t>(value);
}


// ============================================================
// IMU
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

    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    return true;
}


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

    return fabsf(magnitude - 1.0f);
}


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
    sampleBuffer[index][0] = ax;
    sampleBuffer[index][1] = ay;
    sampleBuffer[index][2] = az;
    sampleBuffer[index][3] = gx;
    sampleBuffer[index][4] = gy;
    sampleBuffer[index][5] = gz;
}


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
    Serial.println("Motion detected.");
    Serial.println("Capturing 128 x 6 IMU window...");

    storeSample(
        0,
        firstAx,
        firstAy,
        firstAz,
        firstGx,
        firstGy,
        firstGz
    );

    int sampleIndex = 1;
    unsigned long timeoutStart = millis();

    while (sampleIndex < kWindowSize)
    {
        float ax, ay, az;
        float gx, gy, gz;

        if (
            readIMUSample(
                ax, ay, az,
                gx, gy, gz
            )
        )
        {
            storeSample(
                sampleIndex,
                ax, ay, az,
                gx, gy, gz
            );

            sampleIndex++;
            timeoutStart = millis();
        }

        if (
            millis() - timeoutStart
            >
            2000
        )
        {
            Serial.println("ERROR: IMU capture timeout.");
            return false;
        }
    }

    Serial.println("Capture complete.");
    return true;
}


// ============================================================
// RMS feature extraction
//
// For each channel:
//
// RMS = sqrt(mean(x^2))
//
// Produces exactly 6 features:
// [RMS_ax, RMS_ay, RMS_az, RMS_gx, RMS_gy, RMS_gz]
// ============================================================

void extractRMSFeatures()
{
    for (int channel = 0; channel < kNumChannels; channel++)
    {
        float sumSquares = 0.0f;

        for (int sample = 0; sample < kWindowSize; sample++)
        {
            const float value =
                sampleBuffer[sample][channel];

            sumSquares += value * value;
        }

        rmsFeatures[channel] =
            sqrtf(
                sumSquares
                /
                static_cast<float>(kWindowSize)
            );
    }
}


// ============================================================
// Standardize RMS features and quantize to INT8
//
// Training:
//
// standardized = (RMS - mean) / std
//
// Quantization:
//
// q = standardized / input_scale + zero_point
// ============================================================

bool preprocessRMSAndQuantize()
{
    if (gInput == nullptr)
    {
        return false;
    }

    if (gInput->type != kTfLiteInt8)
    {
        Serial.println("ERROR: model input is not INT8.");
        return false;
    }

    const float inputScale =
        gInput->params.scale;

    const int inputZeroPoint =
        gInput->params.zero_point;

    if (inputScale <= 0.0f)
    {
        Serial.println("ERROR: invalid input scale.");
        return false;
    }

    for (int i = 0; i < kNumFeatures; i++)
    {
        const float stdValue =
            rms_feature_std[i];

        if (fabsf(stdValue) < 1e-9f)
        {
            Serial.println("ERROR: RMS feature std contains zero.");
            return false;
        }

        const float standardized =
            (
                rmsFeatures[i]
                -
                rms_feature_mean[i]
            )
            /
            stdValue;

        const float quantizedFloat =
            standardized
            /
            inputScale
            +
            inputZeroPoint;

        const int32_t quantized =
            static_cast<int32_t>(
                roundf(quantizedFloat)
            );

        gInput->data.int8[i] =
            clampInt8(quantized);
    }

    return true;
}


// ============================================================
// Prediction output
// ============================================================

void printPrediction()
{
    if (gOutput == nullptr)
    {
        return;
    }

    Serial.println();
    Serial.println("========================================");
    Serial.println("GESTURE PREDICTION");
    Serial.println("========================================");

    int bestClass = 0;
    float bestScore = -1.0f;

    for (int i = 0; i < kNumClasses; i++)
    {
        const int8_t raw =
            gOutput->data.int8[i];

        const float score =
            (
                static_cast<int>(raw)
                -
                gOutput->params.zero_point
            )
            *
            gOutput->params.scale;

        Serial.print(kClassLabels[i]);
        Serial.print(": ");
        Serial.println(score, 6);

        if (score > bestScore)
        {
            bestScore = score;
            bestClass = i;
        }
    }

    Serial.println();

    Serial.print("Predicted class: ");
    Serial.println(bestClass);

    Serial.print("Gesture: ");
    Serial.println(kClassLabels[bestClass]);

    Serial.print("Confidence: ");
    Serial.println(bestScore, 6);

    Serial.println("========================================");
}


// ============================================================
// Full RMS pipeline timing
// ============================================================

bool runInference()
{
    const uint32_t featureStart = micros();

    extractRMSFeatures();

    const uint32_t featureEnd = micros();


    const uint32_t preprocessingStart = micros();

    if (!preprocessRMSAndQuantize())
    {
        Serial.println("RMS preprocessing failed.");
        return false;
    }

    const uint32_t preprocessingEnd = micros();


    const uint32_t inferenceStart = micros();

    const TfLiteStatus status =
        gInterpreter->Invoke();

    const uint32_t inferenceEnd = micros();


    if (status != kTfLiteOk)
    {
        Serial.println("Invoke() failed.");
        return false;
    }


    const uint32_t featureUs =
        featureEnd - featureStart;

    const uint32_t preprocessingUs =
        preprocessingEnd - preprocessingStart;

    const uint32_t inferenceUs =
        inferenceEnd - inferenceStart;

    const uint32_t totalUs =
        featureUs
        +
        preprocessingUs
        +
        inferenceUs;


    Serial.println();

    Serial.print("RMS extraction: ");
    Serial.print(featureUs);
    Serial.println(" us");

    Serial.print("Feature preprocessing: ");
    Serial.print(preprocessingUs);
    Serial.println(" us");

    Serial.print("Inference: ");
    Serial.print(inferenceUs);
    Serial.println(" us");

    Serial.print("Inference: ");
    Serial.print(inferenceUs / 1000.0f, 3);
    Serial.println(" ms");

    Serial.print("Total processing: ");
    Serial.print(totalUs);
    Serial.println(" us");

    Serial.print("Total processing: ");
    Serial.print(totalUs / 1000.0f, 3);
    Serial.println(" ms");


    printPrediction();

    return true;
}


// ============================================================
// TensorFlow Lite Micro initialization
// ============================================================

void initializeTensorFlow()
{
    Serial.println("[TFLM] Checking model alignment...");

    const uintptr_t modelAddress =
        reinterpret_cast<uintptr_t>(
            rms_tiny_model
        );

    Serial.print("[TFLM] Model address: 0x");
    Serial.println(modelAddress, HEX);

    Serial.print("[TFLM] Model mod16: ");
    Serial.println(modelAddress % 16);

    if (modelAddress % 16 != 0)
    {
        fatalStop(
            "rms_tiny_model is not 16-byte aligned"
        );
    }


    gModel =
        tflite::GetModel(
            rms_tiny_model
        );

    if (gModel == nullptr)
    {
        fatalStop("GetModel() failed");
    }

    if (
        gModel->version()
        !=
        TFLITE_SCHEMA_VERSION
    )
    {
        fatalStop("TFLite schema mismatch");
    }


    // Tiny Dense INT8 model:
    // FullyConnected layers + Softmax output.

    if (
        gResolver.AddFullyConnected()
        !=
        kTfLiteOk
    )
    {
        fatalStop("AddFullyConnected failed");
    }

    if (
        gResolver.AddSoftmax()
        !=
        kTfLiteOk
    )
    {
        fatalStop("AddSoftmax failed");
    }


    static tflite::MicroInterpreter staticInterpreter(
        gModel,
        gResolver,
        tensorArena,
        kTensorArenaSize
    );

    gInterpreter =
        &staticInterpreter;


    Serial.println("[TFLM] AllocateTensors()...");

    if (
        gInterpreter->AllocateTensors()
        !=
        kTfLiteOk
    )
    {
        fatalStop("AllocateTensors() failed");
    }

    Serial.println("[TFLM] AllocateTensors() PASS");


    Serial.print("[TFLM] Arena used: ");
    Serial.println(
        gInterpreter->arena_used_bytes()
    );


    gInput =
        gInterpreter->input(0);

    gOutput =
        gInterpreter->output(0);


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


    if (gInput->type != kTfLiteInt8)
    {
        fatalStop("Expected INT8 input");
    }

    if (gOutput->type != kTfLiteInt8)
    {
        fatalStop("Expected INT8 output");
    }


    if (gInput->bytes != kNumFeatures)
    {
        fatalStop("Unexpected model input size");
    }

    if (gOutput->bytes != kNumClasses)
    {
        fatalStop("Unexpected model output size");
    }


    Serial.print("[TFLM] Input scale: ");
    Serial.println(
        gInput->params.scale,
        8
    );

    Serial.print("[TFLM] Input zero point: ");
    Serial.println(
        gInput->params.zero_point
    );

    Serial.print("[TFLM] Output scale: ");
    Serial.println(
        gOutput->params.scale,
        8
    );

    Serial.print("[TFLM] Output zero point: ");
    Serial.println(
        gOutput->params.zero_point
    );

    Serial.println("[TFLM] READY");
}


// ============================================================
// IMU initialization
// ============================================================

void initializeIMU()
{
    Serial.println("[IMU] Initializing...");

    if (!IMU.begin())
    {
        fatalStop("IMU.begin() failed");
    }

    Serial.println("[IMU] READY");

    Serial.print("[IMU] Accelerometer sample rate: ");
    Serial.print(
        IMU.accelerationSampleRate()
    );
    Serial.println(" Hz");

    Serial.print("[IMU] Gyroscope sample rate: ");
    Serial.print(
        IMU.gyroscopeSampleRate()
    );
    Serial.println(" Hz");
}


// ============================================================
// Setup
// ============================================================

void setup()
{
    Serial.begin(115200);

    const unsigned long serialStart =
        millis();

    while (
        !Serial
        &&
        (
            millis() - serialStart
            <
            3000
        )
    )
    {
        delay(10);
    }

    delay(500);

    Serial.println();
    Serial.println("========================================");
    Serial.println("RMS TINY INT8");
    Serial.println("REAL-TIME GESTURE CLASSIFIER");
    Serial.println("Arduino Nano 33 BLE");
    Serial.println("========================================");

    initializeTensorFlow();
    initializeIMU();

    Serial.println();
    Serial.println("========================================");
    Serial.println("SYSTEM READY");
    Serial.println("Move the board to perform a gesture.");

    Serial.print("Motion threshold: ");
    Serial.println(
        kMotionThreshold,
        4
    );

    Serial.println("========================================");
    Serial.println();
}


// ============================================================
// Main loop
// ============================================================

void loop()
{
    float ax, ay, az;
    float gx, gy, gz;


    if (
        !readIMUSample(
            ax, ay, az,
            gx, gy, gz
        )
    )
    {
        return;
    }


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


    const float motionScore =
        calculateMotionScore(
            ax, ay, az
        );


    if (
        motionScore
        <
        kMotionThreshold
    )
    {
        return;
    }


    Serial.print("Motion score: ");
    Serial.println(
        motionScore,
        4
    );


    if (
        !captureGestureWindow(
            ax, ay, az,
            gx, gy, gz
        )
    )
    {
        return;
    }


    if (runInference())
    {
        lastGestureTime =
            millis();
    }


    Serial.println();
    Serial.println("Ready for next gesture.");
    Serial.println();
}
