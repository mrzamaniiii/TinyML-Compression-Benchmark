#ifndef STANDARD_CNN_SCALER_H
#define STANDARD_CNN_SCALER_H

constexpr int kRawChannels = 6;

const float raw_mean[kRawChannels] = {
    1.1208049254e-03f,
    -8.2654111087e-02f,
    -1.2643310428e-01f,
    -1.5734744072e+00f,
    1.5375390649e-01f,
    2.9320248961e-01f
};

const float raw_std[kRawChannels] = {
    2.6098391414e-01f,
    6.0880005360e-01f,
    7.4689579010e-01f,
    1.9054401398e+01f,
    1.7840723038e+01f,
    5.6507167816e+01f
};

#endif
