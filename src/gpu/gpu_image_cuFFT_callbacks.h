#ifndef __GPU_IMAGE_CUFFT_CALLBACKS_H__
#define __GPU_IMAGE_CUFFT_CALLBACKS_H__

// cuFFT callbacks
// These must be linked with the static cuFFT library.
// As of cuda 11.4 callbacks are deprecated for separate compilation. These will eventually be replaced with FastFFT.

__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {

    const __half element = ((__half*)dataIn)[offset];
    return (cufftReal)(__half2float(element));
}

__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr);

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr) {
    float scale_factor = *(float*)callerInfo;

    ((cufftComplex*)dataOut)[offset] = (cufftComplex)ComplexScale(element, scale_factor);
}

//__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;
__device__ cufftCallbackStoreC d_scaleFFTAndStorePtr = CB_scaleFFTAndStore;

__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr);

__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr) {

    __half* data_out_half = (__half*)callerInfo;
    //	data_out_half[offset] = __float2half(element);
    //	((cufftReal *)dataOut)[offset] = element;

#ifdef DISABLECACHEHINTS
    ((__half*)data_out_half)[offset] = __float2half(element);
#else
    __stcs(&data_out_half[offset], __float2half(element));
#endif

    //	)_(dataOut)[offset] = __float2half(element);
    //
}

__device__ cufftCallbackStoreR d_mipCCGAndStorePtr = CB_mipCCGAndStore;

template <typename T>
struct CB_complexConjMulLoad_params {
    T*    target;
    float scale;
};

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {
    CB_complexConjMulLoad_params<cufftComplex>* my_params = (CB_complexConjMulLoad_params<cufftComplex>*)callerInfo;
    return (cufftComplex)ComplexConjMulAndScale(my_params->target[offset], ((Complex*)dataIn)[offset], my_params->scale);
    //    return (cufftComplex)make_float2(my_params->scale * __fmaf_rn(my_params->target[offset].x ,  ((Complex *)dataIn)[offset].x, my_params->target[offset].y * ((Complex *)dataIn)[offset].y),
    //    		my_params->scale * __fmaf_rn(my_params->target[offset].y , -((Complex *)dataIn)[offset].x, my_params->target[offset].x * ((Complex *)dataIn)[offset].y));
}

static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {
    CB_complexConjMulLoad_params<__half2>* my_params = (CB_complexConjMulLoad_params<__half2>*)callerInfo;
    return (cufftComplex)ComplexConjMulAndScale((cufftComplex)__half22float2(my_params->target[offset]), ((Complex*)dataIn)[offset], my_params->scale);
}

__device__ cufftCallbackLoadC d_complexConjMulLoad_32f = CB_complexConjMulLoad_32f;
__device__ cufftCallbackLoadC d_complexConjMulLoad_16f = CB_complexConjMulLoad_16f;

typedef struct _CB_realLoadAndClipInto_params {
    int*       mask;
    cufftReal* target;

} CB_realLoadAndClipInto_params;

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {

    CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params*)callerInfo;
    int                            idx       = my_params->mask[offset];
    if ( idx == 0 ) {
        return 0.0f;
    }
    else {
        return my_params->target[idx];
    }
}

__device__ cufftCallbackLoadR d_realLoadAndClipInto = CB_realLoadAndClipInto;

#endif // __GPU_IMAGE_CUFFT_CALLBACKS_H__