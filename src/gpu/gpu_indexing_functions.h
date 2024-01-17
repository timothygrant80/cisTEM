#ifndef _SRC_GPU_GPU_INDEXING_FUNCTIONS_H_
#define _SRC_GPU_GPU_INDEXING_FUNCTIONS_H_

/**
 * @brief Get a one-dimensional index for the block in the grid.
 * 
 * @return blockIdx.x + gridDim.x * blockIdx.y;
 */
__device__ __forceinline__ int LinearBlockIdx_2dGrid( ) {
    return blockIdx.x + gridDim.x * blockIdx.y;
}

/**
 * @brief Get a one-dimensional index for the block in the grid. Could 
 * 
 * @return blockIdx.x + gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);
 */
__device__ __forceinline__ int LinearBlockIdx_3dGrid( ) {
    return blockIdx.x + gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);
}

/**
 * @brief Get a one-dimensional index for trhead in the 2d in block.
 * 
 * @return threadIdx.x + threadIdx.y * blockDim.x;
 */
__device__ __forceinline__ int LinearThreadIdxInBlock_2dGrid( ) {
    return threadIdx.x + threadIdx.y * blockDim.x;
}

/**
 * @brief Get a one-dimensional index for trhead in the 3d in block.
 * 
 * @return threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y);
 */
__device__ __forceinline__ int LinearThreadIdxInBlock_3dGrid( ) {
    return threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y);
}

/**
 * @brief The number of threads in a grid (launched) used for 1d grid stride loops. could rename NumThreads_1dGrid.
 * 
 * @return return gridDim.x * blockDim.x;
 */
__device__ __forceinline__ int GridStride_1dGrid( ) {
    return gridDim.x * blockDim.x;
}

/**
 * @brief The number of threads in a grid (launched) used for 1d grid stride loops blockDim.y > 1 Could rename NumThreads_2dGrid.
 * For example, when you want bounds checking with x/y but a single level loop is sufficient
 * 
 * @return gridDim.x * blockDim.x + gridDim.y * blockDim.y;
 */
__device__ __forceinline__ int GridStride_2dGrid( ) {
    return gridDim.x * blockDim.x + gridDim.y * blockDim.y;
}

/**
 * @brief The number of x-threads in a grid (launched) used for 2d grid stride loops. Could rename NumThreads_2dGrid.
 * 
 * @return gridDim.x * blockDim.x;
 */
__device__ __forceinline__ int GridStride_2dGrid_X( ) {
    return gridDim.x * blockDim.x;
}

/**
 * @brief The number of y-threads in a grid (launched) used for 2d grid stride loops. Could rename NumThreads_2dGrid_Y.
 * 
 * @return gridDim.y * blockDim.y;
 */
__device__ __forceinline__ int GridStride_2dGrid_Y( ) {
    return gridDim.y * blockDim.y;
}

/**
 * @brief The number of blocks in a grid. Could rename NumBlocks_2dGrid.
 * 
 * @return gridDim.x * gridDim.y;
 */
__device__ __forceinline__ int GridDimension_2d( ) {
    return gridDim.x * gridDim.y;
}

/**
 * @brief The number of blocks in a grid. Could rename NumBlocks_3dGrid.
 * 
 * @return return gridDim.x * gridDim.y * gridDim.z;
 */
__device__ __forceinline__ int GridDimension_3d( ) {
    return gridDim.x * gridDim.y * gridDim.z;
}

/**
 * @brief The number of threads in a grid. Could rename NumThreads_2dGrid.
 * 
 * @return blockDim.x * blockDim.y;
 */
__device__ __forceinline__ int BlockDimension_2d( ) {
    return blockDim.x * blockDim.y;
}

/**
 * @brief The number of threads in a grid. Could rename NumThreads_3dGrid.
 * 
 * @return blockDim.x * blockDim.y * blockDim.z;
 */
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

/**
*
 * @return blockIdx.x * blockDim.x + threadIdx.x
 */
__device__ __forceinline__ int physical_X( ) {
    return physical_X_3d_grid( );
}

/**
 * @return blockIdx.y * blockDim.y + threadIdx.y
 
*/
__device__ __forceinline__ int physical_Y( ) {
    return physical_Y_3d_grid( );
}

/**
 * @return blockIdx.z * blockDim.z + threadIdx.z
*/
__device__ __forceinline__ int physical_Z( ) {
    return physical_Z_3d_grid( );
}

#endif