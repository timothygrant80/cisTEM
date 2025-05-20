#ifndef _SRC_GPU_CORE_EXTENSIONS_DATA_VIEWS_POINTERS_H_
#define _SRC_GPU_CORE_EXTENSIONS_DATA_VIEWS_POINTERS_H_

#include "../../gpu_core_headers.h"

// Manage an array of pointers to device memory.
// TODO: I don't remember why I (thought) this was needed over a std::vector.
// Probably related to copying or the custome allocator, which could probably be replaced with a std::vector.

template <typename PtrType>
class DevicePointerArray {
  public:
    DevicePointerArray( )
        : _capacity(0), _size(0){ };

    DevicePointerArray(int wanted_capacity)
        : _capacity(wanted_capacity), _size(0) {
        Allocate( );
    }

    // Prevent copying.
    DevicePointerArray(const DevicePointerArray&)            = delete;
    DevicePointerArray& operator=(const DevicePointerArray&) = delete;

    ~DevicePointerArray( ) { Deallocate( ); }

    inline void Allocate( ) {
        MyDebugAssertTrue(_capacity > 0, "Cannot allocate an array of size 0.");
        MyDebugAssertFalse(_size > 0, "Cannot allocate an array that is already allocated.");
        cudaErr(cudaMallocManaged(&ptr_array, _capacity * sizeof(PtrType*)));
    }

    inline void Deallocate( ) {
        if ( _size > 0 ) {
            // Make sure any non-owned data are left alone.
            for ( int i = 0; i < _size; i++ ) {
                ptr_array[i] = nullptr;
            }
            _size = 0;
            cudaErr(cudaFree(ptr_array));
        }

        return;
    }

    // Set the array to already allocated pointers for some other memory.
    inline void SetPointer(PtrType* data_ptr, int index) {
        // We could dynamically grow this, but for now, we just assert it is not too big.
        MyDebugAssertFalse(index >= _capacity, "Index out of bounds.");
        ptr_array[index] = data_ptr;
        return;
    }

    // Return the size of the array of pointers.
    inline int size( ) const { return _size; }

    inline int capacity( ) const { return _capacity; }

    void resize(int wanted_size) {
        if ( _capacity == 0 ) {
            _capacity = wanted_size;
            Allocate( );
        }
        else {
            // We could dynamically grow this, but for now, we just assert it is not too big.
            MyDebugAssertTrue(wanted_size == _capacity, "Cannot resize to a size larger than the capacity.");
        }
        return;
    }

    // We want to be able to share this pointer with other objects.
    PtrType** ptr_array;

  private:
    int _size;
    int _capacity;
};

#endif
