// #include "gpu_core_headers.h"

// #include "TensorManager.h"

// // #define N_CACHE_LINES 32

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {

//     cuTensorErr(cutensorInit(&handle));
//     SetDefaultValues( );

// #ifdef N_CACHE_LINES
//     // Set number of cache lines
//     constexpr int32_t numCachelines = N_CACHE_LINES;
//     // Set cache size and allocate
//     const size_t             sizeCache  = numCachelines * sizeof(cutensorPlanCacheline_t);
//     cutensorPlanCacheline_t* cachelines = (cutensorPlanCacheline_t*)malloc(sizeCache);
//     // Attach cache
//     cuTensorErr(cutensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines));
// #endif
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager(const GpuImage& wanted_props) {
//     // Set all the properties of the tensor manager based on the reference GpuImage.
//     cuTensorErr(cutensorInit(&handle));
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::~TensorManager( ) {
//     Deallocate( );
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::Deallocate( ) {

// #ifdef N_CACHE_LINES
//     // Detach cache and free-up resources
//     cuTensorErr(cutensorHandleDetachPlanCachelines(&handle));
// #endif
//     // By design the TensorManager is non-owning, so the only thing to free is the workspace if it is allocated
//     if ( workspace_ptr ) {
//         cudaErr(cudaFree(workspace_ptr));
//         workspace_ptr = nullptr;
//     }
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// template <class ArrayType>
// bool TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::CheckForSetMetaData(ArrayType& is_property_set) {

//     for ( int i = 0; i < cistem::gpu::max_tensor_manager_tensors; i++ ) {
//         if ( is_tensor_active[i] && ! is_property_set[i] ) {
//             wxPrintf("\nTensor %c is active but not all properties are set.\n", cistem::gpu::tensor_id::tensor_names[i]);
//             return false;
//             break;
//         }
//     }
//     return true;
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetDefaultValues( ) {
//     alpha = ComputeType(0.f);
//     beta  = ComputeType(0.f);

//     workspace_size = 0;
//     workspace_ptr  = nullptr;

//     is_set_operation = false;

//     std::size_t i;

//     for ( i = 0; i != is_tensor_allocated.size( ); ++i ) {
//         is_tensor_allocated[i] = false;
//     }

//     for ( i = 0; i != is_tensor_active.size( ); ++i ) {
//         is_tensor_active[i] = false;
//     }

//     for ( i = 0; i != is_set_tensor_descriptor.size( ); ++i ) {
//         is_set_tensor_descriptor[i] = false;
//     }

//     for ( i = 0; i != is_set_unary_operator.size( ); ++i ) {
//         is_set_unary_operator[i] = false;
//     }

//     for ( i = 0; i != is_set_n_elements_in_each_tensor.size( ); ++i ) {
//         is_set_n_elements_in_each_tensor[i] = false;
//     }

//     for ( i = 0; i != n_elements_in_each_tensor.size( ); ++i ) {
//         n_elements_in_each_tensor[i] = 1;
//     }

//     for ( i = 0; i != is_set_n_elements_in_each_tensor.size( ); ++i ) {
//         is_set_n_elements_in_each_tensor[i] = false;
//     }
// }

// // template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// // template <class ThisType>
// // void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetTensorCudaType(TensorID tid) {

// //     if ( std::is_same_v<ThisType, float2> )
// //         tensor_cuda_types[tid] = CUDA_C_32F;
// //     return;

// //     if ( std::is_same_v<ThisType, nv_bfloat16> )
// //         tensor_cuda_types[tid] = CUDA_R_16BF;
// //     return;

// //     if ( std::is_same_v<ThisType, nv_bfloat162> )
// //         tensor_cuda_types[tid] = CUDA_C_16BF;
// //     return;

// //     if ( std::is_same_v<ThisType, __half> )
// //         tensor_cuda_types[tid] = CUDA_R_16F;
// //     return;

// //     if ( std::is_same_v<ThisType, __half2> )
// //         tensor_cuda_types[tid] = CUDA_C_16F;
// //     return;

// //     // If we got here there is a problem.

// //     std::cerr << "Error: TensorManager::SetTensorCudaType: Unsupported type.\n";
// //     exit(1);
// // }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetTensorDescriptors( ) {

//     MyDebugAssertTrue(CheckForSetMetaData(is_set_n_modes), "Set n_modes before setting tensor descriptors");
//     MyDebugAssertTrue(CheckForSetMetaData(is_set_extents_of_each_tensor), "Set extents of each tensor before setting tensor descriptors");
//     MyDebugAssertTrue(CheckForSetMetaData(is_set_unary_operator), "Set unary operator before setting tensor descriptors");

//     for ( std::size_t i = 0; i != tensor_descriptor.size( ); ++i ) {
//         if ( is_tensor_active[i] ) {
//             // std::cerr << "Trying to set descriptor for tensor " << cistem::gpu::tensor_id::tensor_names[i] << "\n";
//             // std::cerr << "n_modes: " << n_modes[i] << "\n";
//             // for ( auto v : extents_of_each_tensor[i] ) {
//             //     std::cerr << "extent is " << v << "\n";
//             // }
//             // std::cerr << "Cuda data type is " << tensor_cuda_types[i] << "\n";
//             // std::cerr << "unary operator is " << unary_operator[i] << "\n";
//             cutensorStatus_t err = cutensorInitTensorDescriptor(&handle,
//                                                                 &tensor_descriptor[i],
//                                                                 n_modes[i],
//                                                                 extents_of_each_tensor[i].data( ),
//                                                                 NULL /* stride assuming a packed layout including FFTW padding*/,
//                                                                 GetCudaDataType(TensorID(i)),
//                                                                 unary_operator[i]);

//             cuTensorErr(CUTENSOR_STATUS_SUCCESS); // Only matters in debug mode;
//             if ( err == CUTENSOR_STATUS_SUCCESS ) {
//                 is_set_tensor_descriptor[i] = true; // if not true will trigger other asserts but should be redundant at this point in the logic.
//             }
//         }
//     }
// }

// template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::GetWorkSpaceSize( ) {

//     MyDebugAssertTrue(is_set_operation, "Set operations before getting work space size");
//     workspace_size = 0;
//     if ( workspace_ptr ) {
//         cudaErr(cudaFree(workspace_ptr));
//         workspace_ptr = nullptr;
//     }

//     switch ( operation ) {
//         case TensorOP::reduction: {
//             MyDebugAssertTrue(CheckForSetMetaData(is_set_tensor_descriptor), "Set tensor descriptor before getting work space size");
//             MyDebugAssertTrue(CheckForSetMetaData(is_tensor_allocated), "Set tensor allocated before getting work space size");
//             if ( is_tensor_active[TensorID::A] && is_tensor_active[TensorID::B] && ! is_tensor_active[TensorID::C] ) {

//                 cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
//                                                               my_ptrs._a_ptr, &tensor_descriptor[TensorID::A], modes[TensorID::A].data( ),
//                                                               my_ptrs._b_ptr, &tensor_descriptor[TensorID::B], modes[TensorID::B].data( ),
//                                                               my_ptrs._b_ptr, &tensor_descriptor[TensorID::B], modes[TensorID::B].data( ),
//                                                               cutensor_op, my_types._compute_type, &workspace_size));
//             }
//             else if ( is_tensor_active[TensorID::A] && is_tensor_active[TensorID::B] && is_tensor_active[TensorID::C] ) {
//                 cuTensorErr(cutensorReductionGetWorkspaceSize(&handle,
//                                                               my_ptrs._a_ptr, &tensor_descriptor[TensorID::A], modes[TensorID::A].data( ),
//                                                               my_ptrs._b_ptr, &tensor_descriptor[TensorID::B], modes[TensorID::B].data( ),
//                                                               my_ptrs._c_ptr, &tensor_descriptor[TensorID::C], modes[TensorID::C].data( ),
//                                                               cutensor_op, my_types._compute_type, &workspace_size));
//             }
//             else {
//                 wxPrintf("Active tensors expected to be A and B, or A and B and C.\n");
//                 PrintActiveTensorNames( );
//                 wxPrintf("Error: TensorManager::GetWorkSpaceSize: Unsupported operation.\n\n");
//                 wxSleep(2);
//                 DEBUG_ABORT;
//             }
//             break;
//         }
//         case TensorOP::contraction: {
//             MyDebugAssertTrue(false, "Contraction not implemented");
//             break;
//         }
//         case TensorOP::binary: {
//             MyDebugAssertTrue(false, "Binary not implemented");
//             break;
//         }
//         case TensorOP::ternary: {
//             MyDebugAssertTrue(false, "Ternary not implemented");
//             break;
//         }
//         default: {
//             wxPrintf("Unknown operation type in TensorManager::GetWorkSpaceSize\n");
//             wxSleep(2);
//             DEBUG_ABORT;
//             break;
//         }
//     }

//     if ( workspace_size > 0 ) {
//         if ( cudaSuccess != cudaMalloc(&workspace_ptr, workspace_size) ) {
//             workspace_ptr  = nullptr;
//             workspace_size = 0;
//         }
//     }
// }

// // template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// // template <char Mode>
// // void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetModes(TensorID tid) {
// //     modes[tid].push_back(Mode);
// //     n_modes[tid]        = modes[tid].size( );
// //     is_set_modes[tid]   = true;
// //     is_set_n_modes[tid] = true;
// //     // MyDebugAssert(modes[tid].size( ) <= cistem::gpu::max_tensor_manager_dimensions, "Too many modes for a given tensor ID.");
// //     extent_of_each_mode.try_emplace(Mode, 0); // This will be checked later for proper setting, but don't overwrite if it already exists
// //     is_tensor_active[tid] = true;
// // };

// // template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// // template <char Mode, char... OtherModes>
// // void TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::SetModes(TensorID tid) {
// //     modes[tid].push_back(Mode);
// //     SetModes<OtherModes...>(tid);
// // };

// // template <class TypeA, class TypeB, class TypeC, class TypeD, class ComputeType>
// // TensorManager<TypeA, TypeB, TypeC, TypeD, ComputeType>::TensorManager( ) {
// // }

// // So we can do separate compilation
// template class TensorManager<float, float, float, float, float>;
// template class TensorManager<nv_bfloat16, nv_bfloat16, nv_bfloat16, nv_bfloat16, nv_bfloat16>;
// template class TensorManager<__half, __half, __half, __half, __half>;

// // template void TensorManager<float, float, float, float, float>::SetModes<char>(TensorID tid);
// // template void TensorManager<float, float, float, float, float>::SetModes<char, char>(TensorID tid);
// // template void TensorManager<float, float, float, float, float>::SetModes<char, char, char>(TensorID tid);
// // template void TensorManager<float, float, float, float, float>::SetModes<char, char, char, char>(TensorID tid);
