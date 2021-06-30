#ifndef PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_

struct ConvFuse {
    ConvFuse() {
        ifReLU = false;
        ifEltwise = false;
        ifEltwiseReLU = false;
        ifConcat = false;
        ifPReLU = false;
        ifReLUMax = false;
        ifEltwiseReLUMax = false;
        ifEltwisePReLU = false;
        reluMax = 0.0f;
        eltwiseReluMax = 0.0f;
        concatOffset = 0;
        concatStride = 0;
        preDataGrp = nullptr;
        concatOutData = nullptr;
        negeData = nullptr;
        negeEltData = nullptr;
    }

    bool ifReLU;
    bool ifEltwise;
    bool ifEltwiseReLU;
    bool ifConcat;
    bool ifPReLU;
    bool ifReLUMax;
    bool ifEltwiseReLUMax;
    bool ifEltwisePReLU;
    float reluMax;
    float eltwiseReluMax;
    int concatOffset;
    int concatStride;
    void* preDataGrp;
    void* concatOutData;
    void* negeData;
    void* negeEltData;
};

#endif //PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_