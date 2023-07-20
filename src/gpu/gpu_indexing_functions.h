#ifndef _SRC_GPU_GPU_INDEXING_FUNCTIONS_H_
#define _SRC_GPU_GPU_INDEXING_FUNCTIONS_H_

__device__ __forceinline__ int LinearBlockIdx_2dGrid( ) {
    return blockIdx.x + gridDim.x * blockIdx.y;
}

__device__ __forceinline__ int LinearBlockIdx_3dGrid( ) {
    return blockIdx.x + gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);
}

__device__ __forceinline__ int LinearThreadIdxInBlock_2dGrid( ) {
    return threadIdx.x + threadIdx.y * blockDim.x;
}

__device__ __forceinline__ int LinearThreadIdxInBlock_3dGrid( ) {
    return threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y);
}

__device__ __forceinline__ int GridStride_1dGrid( ) {
    return gridDim.x * blockDim.x;
}

__device__ __forceinline__ int GridStride_2dGrid( ) {
    return gridDim.x * blockDim.x + gridDim.y * blockDim.y;
}

__device__ __forceinline__ int GridDimension_2d( ) {
    return gridDim.x * gridDim.y;
}

__device__ __forceinline__ int GridDimension_3d( ) {
    return gridDim.x * gridDim.y * gridDim.z;
}

__device__ __forceinline__ int BlockDimension_2d( ) {
    return blockDim.x * blockDim.y;
}

__device__ __forceinline__ int BlockDimension_3d( ) {
    return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ int physical_X_3d_grid( ) {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int physical_Y_3d_grid( ) {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

__device__ __forceinline__ int physical_Z_3d_grid( ) {
    return blockIdx.z * blockDim.z + threadIdx.z;
}

__device__ __forceinline__ int physical_X_2d_grid( ) {
    return physical_X_3d_grid( );
}

__device__ __forceinline__ int physical_Y_2d_grid( ) {
    return physical_Y_3d_grid( );
}

__device__ __forceinline__ int physical_Z_2d_grid( ) {
    return blockIdx.z;
}

__device__ __forceinline__ int physical_X_1d_grid( ) {
    return physical_X_3d_grid( );
}

__device__ __forceinline__ int physical_Y_1d_grid( ) {
    return blockIdx.y;
}

__device__ __forceinline__ int physical_Z_1d_grid( ) {
    return blockIdx.z;
}

// In some cases, having the explicit name can clarify the code.
// If not, alias the most generic method.

__device__ __forceinline__ int LinearBlockIdx( ) {
    return LinearBlockIdx_3dGrid( );
}

__device__ __forceinline__ int BlockDimension( ) {
    return BlockDimension_3d( );
}

__device__ __forceinline__ int GridDimension( ) {
    return GridDimension_3d( );
}

__device__ __forceinline__ int physical_X( ) {
    return physical_X_3d_grid( );
}

__device__ __forceinline__ int physical_Y( ) {
    return physical_Y_3d_grid( );
}

__device__ __forceinline__ int physical_Z( ) {
    return physical_Z_3d_grid( );
}

#endif