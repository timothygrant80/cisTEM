#ifndef __INCLUDE_DETAIL_DEVICE_PROPERTIES_H__
#define __INCLUDE_DETAIL_DEVICE_PROPERTIES_H__

namespace FastFFT {
typedef struct __align__(32) _DeviceProps {
    int device_id;
    int device_arch;
    int max_shared_memory_per_block;
    int max_shared_memory_per_SM;
    int max_registers_per_block;
    int max_persisting_L2_cache_size;
}

DeviceProps;

} // namespace FastFFT

#endif // __INCLUDE_DETAIL_DEVICE_PROPERTIES_H__