/*
Provide an interface to the cuTensor library to the cistem GpuImage class
*/

#ifndef _SRC_GPU_TENSORMANAGER_H_
#define _SRC_GPU_TENSORMANAGER_H_

#include <cutensor.h>
#include <cistem_config.h>
#include "../constants/constants.h"

class GpuImage;

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename TypeCompute>
struct TensorTypes {
    // Use this to catch unsupported input/ compute types and throw exception.
    bool _a_type       = false;
    bool _b_type       = false;
    bool _c_type       = false;
    bool _d_type       = false;
    bool _compute_type = false;
};

// Input real, compute single-precision
template <>
struct TensorTypes<float, float, float, float, float> {
    cudaDataType_t        _a_type       = CUDA_R_32F;
    cudaDataType_t        _b_type       = CUDA_R_32F;
    cudaDataType_t        _c_type       = CUDA_R_32F;
    cudaDataType_t        _d_type       = CUDA_R_32F;
    cutensorComputeType_t _compute_type = CUTENSOR_COMPUTE_32F;
};

template <>
struct TensorTypes<nv_bfloat16, nv_bfloat16, nv_bfloat16, nv_bfloat16, nv_bfloat16> {
    cudaDataType_t        _a_type       = CUDA_R_16BF;
    cudaDataType_t        _b_type       = CUDA_R_16BF;
    cudaDataType_t        _c_type       = CUDA_R_16BF;
    cudaDataType_t        _d_type       = CUDA_R_16BF;
    cutensorComputeType_t _compute_type = CUTENSOR_COMPUTE_16BF;
};

template <>
struct TensorTypes<__half, __half, __half, __half, __half> {
    cudaDataType_t        _a_type       = CUDA_R_16F;
    cudaDataType_t        _b_type       = CUDA_R_16F;
    cudaDataType_t        _c_type       = CUDA_R_16F;
    cudaDataType_t        _d_type       = CUDA_R_16F;
    cutensorComputeType_t _compute_type = CUTENSOR_COMPUTE_16F;
};

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD>
struct TensorPtrs {
    // Use this to catch unsupported input/ compute types and throw exception.

    int* _a_ptr = nullptr;
    int* _b_ptr = nullptr;
    int* _c_ptr = nullptr;
    int* _d_ptr = nullptr;
};

// Input real, compute single-precision
template <>
struct TensorPtrs<float, float, float, float> {
    float* _a_ptr;
    float* _b_ptr;
    float* _c_ptr;
    float* _d_ptr;
};

template <>
struct TensorPtrs<__half, __half, __half, __half> {
    __half* _a_ptr;
    __half* _b_ptr;
    __half* _c_ptr;
    __half* _d_ptr;
};

template <>
struct TensorPtrs<nv_bfloat16, nv_bfloat16, nv_bfloat16, nv_bfloat16> {
    nv_bfloat16* _a_ptr;
    nv_bfloat16* _b_ptr;
    nv_bfloat16* _c_ptr;
    nv_bfloat16* _d_ptr;
};

using TensorID = cistem::gpu::tensor_id::Enum;
using TensorOP = cistem::gpu::tensor_op::Enum;

template <class TypeA, class TypeB, class TypeC, class TypeD, class TypeCompute>
class TensorManager {

  public:
    TensorManager( );
    TensorManager(const GpuImage& wanted_props);
    ~TensorManager( );

    void Deallocate( );

    cutensorHandle_t handle;

    void SetDefaultValues( );

    // template <class ThisType>
    // void SetTensorCudaType(TensorID tid);

    template <class WantedPtr>
    inline bool CheckPtrType(TensorID tid, WantedPtr* wanted_ptr) {
        switch ( tid ) {
            case TensorID::A:
                return std::is_same_v<WantedPtr*, decltype(my_ptrs._a_ptr)>;
            case TensorID::B:
                return std::is_same_v<WantedPtr*, decltype(my_ptrs._b_ptr)>;
            case TensorID::C:
                return std::is_same_v<WantedPtr*, decltype(my_ptrs._c_ptr)>;
            case TensorID::D:
                return std::is_same_v<WantedPtr*, decltype(my_ptrs._d_ptr)>;
            default:
                static_assert("TensorID not supported");
        }
    };

    template <class ThisType>
    inline void SetTensorPointers(TensorID tid, ThisType* ptr) {
        MyDebugAssertTrue(CheckPtrType<ThisType>(tid, ptr), "Type mismatch in pointer assignment");

        switch ( tid ) {
            case TensorID::A:
                my_ptrs._a_ptr = ptr;
                break;
            case TensorID::B:
                my_ptrs._b_ptr = ptr;
                break;
            case TensorID::C:
                my_ptrs._c_ptr = ptr;
                break;
            case TensorID::D:
                my_ptrs._d_ptr = ptr;
                break;
            default:
                static_assert("TensorID not supported");
        }
    }

    void GetWorkSpaceSize( );

    inline void SetAlphaAndBeta(TypeCompute wanted_alpha, TypeCompute wanted_beta) {
        alpha = wanted_alpha;
        beta  = wanted_beta;
    };

    inline void SetExtent(char mode, int64_t wanted_extent) {
        auto search = extent_of_each_mode.find(mode);
        if ( search != extent_of_each_mode.end( ) ) {
            extent_of_each_mode[mode] = wanted_extent;
        }
        else {
            wxPrintf("Found the following modes: ");
            for ( auto m : extent_of_each_mode ) {
                wxPrintf("%c, %d ", char(m.first), int(m.second));
            }
            wxPrintf("\n");
            wxPrintf("mode requested %c for extent %i\n", mode, int(wanted_extent));
            MyAssertTrue(false, "Could not find mode in wanted_extent");
        }
    };

    inline int64_t GetExtent(char mode) {
        auto search = extent_of_each_mode.find(mode);
        if ( search != extent_of_each_mode.end( ) ) {
            return extent_of_each_mode[mode];
        }
        else {
            wxPrintf("mode requested %c\n", mode);

            MyAssertTrue(false, "Could not find mode in wanted_extent");
        }
    };

    template <char Mode>
    void SetModes(TensorID tid) {
        // wxPrintf("Setting mode %c, for tensor %i\n", Mode, int(tid));
        modes[tid].push_back(Mode);
        n_modes[tid]        = modes[tid].size( );
        is_set_modes[tid]   = true;
        is_set_n_modes[tid] = true;
        // MyDebugAssert(modes[tid].size( ) <= cistem::gpu::max_tensor_manager_dimensions, "Too many modes for a given tensor ID.");
        extent_of_each_mode.try_emplace(Mode, 0); // This will be checked later for proper setting, but don't overwrite if it already exists
        is_tensor_active[tid] = true;
    };

    template <char Mode, char... OtherModes>
    void SetModes(TensorID tid) {
        // wxPrintf("Setting mode %c, for tensor %i\n", Mode, int(tid));

        modes[tid].push_back(Mode);
        extent_of_each_mode.try_emplace(Mode, 0); // This will be checked later for proper setting, but don't overwrite if it already exists
        SetModes<OtherModes...>(tid);
    };

    // template <char Mode>
    // void SetModes(TensorID tid);

    // template <char Mode, char... OtherModes>
    // void SetModes(TensorID tid);

    void SetTensorDescriptors( );

    inline void SetExtentOfTensor(TensorID tid) {
        MyDebugAssertTrue(is_set_modes[tid], "Tensor ID not set.");
        MyDebugAssertTrue(is_set_n_modes[tid], "Number of modes not set.");
        for ( auto mode : modes[tid] )
            extents_of_each_tensor[tid].push_back(extent_of_each_mode[mode]);

        MyDebugAssertTrue(extents_of_each_tensor[tid].size( ) == n_modes[tid], "Number of extents not equal to number of modes.");
        is_set_extents_of_each_tensor[tid] = true;
    }

    inline void SetNElementsInEachTensor(TensorID tid) {
        MyDebugAssertTrue(is_set_extents_of_each_tensor[tid], "Extents of each tensor not set.");
        MyDebugAssertFalse(is_set_n_elements_in_each_tensor[tid], "Number of elements in each tensor already set.");
        for ( auto extent : extents_of_each_tensor[tid] )
            n_elements_in_each_tensor[tid] *= extent;

        is_set_n_elements_in_each_tensor[tid] = true;
    }

    inline void SetNElementsForAllActiveTensors( ) {
        for ( int tid = 0; tid < cistem::gpu::max_tensor_manager_tensors; tid++ ) {
            if ( is_tensor_active[tid] )
                SetNElementsInEachTensor(TensorID(tid));
        }
    }

    inline size_t GetNElementsInTensor(TensorID tid) {
        MyDebugAssertTrue(is_tensor_active[tid], "Tensor ID not active.");
        MyDebugAssertTrue(is_set_n_elements_in_each_tensor[tid], "Number of elements in each tensor not set.");
        return n_elements_in_each_tensor[tid];
    }

    inline void SetUnaryOperator(TensorID tid, cutensorOperator_t wanted_unary_op) {
        MyDebugAssertTrue(is_tensor_active[tid], "Tensor ID not active.");
        unary_operator[tid]        = wanted_unary_op;
        is_set_unary_operator[tid] = true;
        // wxPrintf("Set unary operator for tensor %i\n", int(tid));
    };

    void PrintActiveTensorNames( ) {
        for ( auto tid : is_tensor_active ) {
            std::cerr << "Tensor " << cistem::gpu::tensor_id::tensor_names[tid] << " is " << (is_tensor_active[tid] ? "active" : "not active") << "\n";
        }
    }

    inline void TensorIsAllocated(TensorID allocated_id) {
        MyDebugAssertTrue(is_tensor_active[allocated_id], "Tensor ID not active.");
        is_tensor_allocated[allocated_id] = true;
    }

    inline void SetTensorOperation(TensorOP wanted_op) {
        MyDebugAssertFalse(is_set_operation, "Tensor operation already set.");
        operation = wanted_op;
        // TODO: set this in another function with case
        cutensor_op      = CUTENSOR_OP_MAX;
        is_set_operation = true;
        std::cerr << "Set tensor operation to " << operation << "\n";
    }

    std::array<std::vector<int32_t>, cistem::gpu::max_tensor_manager_tensors> modes;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                 is_set_modes;

    std::array<int32_t, cistem::gpu::max_tensor_manager_tensors> n_modes;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>    is_set_n_modes;

    std::unordered_map<int32_t, int64_t> extent_of_each_mode;

    std::array<std::vector<int64_t>, cistem::gpu::max_tensor_manager_tensors> extents_of_each_tensor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                 is_set_extents_of_each_tensor;

    std::array<cutensorTensorDescriptor_t, cistem::gpu::max_tensor_manager_tensors> tensor_descriptor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>                       is_set_tensor_descriptor;

    std::array<cutensorOperator_t, cistem::gpu::max_tensor_manager_tensors> unary_operator;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>               is_set_unary_operator;

    std::array<size_t, cistem::gpu::max_tensor_manager_tensors> n_elements_in_each_tensor;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors>   is_set_n_elements_in_each_tensor;

    std::array<bool, cistem::gpu::max_tensor_manager_tensors> is_tensor_allocated;
    std::array<bool, cistem::gpu::max_tensor_manager_tensors> is_tensor_active;

    TypeCompute alpha;
    TypeCompute beta;

    template <class ArrayType>
    bool CheckForSetMetaData(ArrayType& is_property_set);

    // We could have multiple operations handled by the same tensor manager.
    TensorOP           operation;
    bool               is_set_operation;
    cutensorOperator_t cutensor_op;

    uint64_t workspace_size;
    bool     is_set_workspace_size;

    void* workspace_ptr;

    TensorTypes<TypeA, TypeB, TypeC, TypeD, TypeCompute> my_types;
    TensorPtrs<TypeA, TypeB, TypeC, TypeD>               my_ptrs;

    inline cudaDataType_t GetCudaDataType(TensorID tid) {
        switch ( tid ) {
            case TensorID::A:
                return my_types._a_type;
            case TensorID::B:
                return my_types._b_type;
            case TensorID::C:
                return my_types._c_type;
            case TensorID::D:
                return my_types._d_type;
            default:
                MyDebugAssertTrue(false, "Invalid tensor ID.");
        }
    }
};
#endif