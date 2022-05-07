

class DeviceManager {

  public:
    int  nGPUs;
    int  gpuIDX;
    bool is_manager_initialized = false;

    DeviceManager( );
    DeviceManager(int wanted_number_of_gpus);
    virtual ~DeviceManager( );

    void Init(int wanted_number_of_gpus);
    void SetGpu(int cpu_thread_idx);
    void ResetGpu( );
    void ListDevices( );

  private:
};
