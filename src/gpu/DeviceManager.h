#ifndef _SRC_GPU_DEVICEMANAGER_H_
#define _SRC_GPU_DEVICEMANAGER_H_

class DeviceManager {

  public:
    int  nGPUs;
    int  gpuIDX;
    bool is_manager_initialized = false;

    DeviceManager( );
    ~DeviceManager( );

    void Init(int wanted_number_of_gpus, MyApp* parent_ptr);
    void SetGpu( );
    void ResetGpu( );
    void ListDevices( );

  private:
};

#endif // _SRC_GPU_DEVICEMANAGER_H_
