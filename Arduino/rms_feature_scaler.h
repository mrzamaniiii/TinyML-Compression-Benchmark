#ifndef RMS_FEATURE_SCALER_H
#define RMS_FEATURE_SCALER_H

constexpr int kRmsFeatures = 6;

const float rms_feature_mean[kRmsFeatures] = {
    2.0916862786e-01f,
    4.7709774971e-01f,
    6.5320366621e-01f,
    1.5720499039e+01f,
    1.2809416771e+01f,
    4.1788352966e+01f
};

const float rms_feature_std[kRmsFeatures] = {
    1.5607988834e-01f,
    3.8711392879e-01f,
    3.8362446427e-01f,
    1.0880880356e+01f,
    1.2419378281e+01f,
    3.8039154053e+01f
};

#endif
