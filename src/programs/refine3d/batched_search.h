#ifndef _SRC_PROGRAMS_REFINE3D_BATCHED_SEARCH_H_
#define _SRC_PROGRAMS_REFINE3D_BATCHED_SEARCH_H_

class GpuImage;

// class Peak;

typedef __align__(16) struct _IntegerPeak {
    float value;
    int   z;
    int   physical_address_within_image;
} IntegerPeak;

class BatchedSearch {

  public:
    BatchedSearch( );

    ~BatchedSearch( );

    void Deallocate( );

    int  index;
    void Init(GpuImage& reference_image, int wanted_number_search_images, int wanted_batch_size, bool test_mirror, int max_pix_x = 0, int max_pix_y = 0);

    void SetMaxSearchExtension(int max_pixel_radius_x, int max_pixel_radius_y) {
        _max_pixel_radius_x = max_pixel_radius_x;
        _max_pixel_radius_y = max_pixel_radius_y;
    }

    void SetMinSearchExtension(int wanted_min_pixel_radius_x_y) {
        _min_pixel_radius_x_y = wanted_min_pixel_radius_x_y;
    }

    inline int is_initialized( ) const {
        return _is_initialized;
    }

    void SetDeviceBuffer( );

    inline int n_search_images( ) const { return _n_search_images; }

    inline int n_batches( ) const { return _n_batches; };

    inline int n_images_in_this_batch( ) const { return (index < _n_batches - 1) ? _batch_size : _n_in_last_batch; };

    inline int stride( ) const { return _stride; };

    inline int max_pixel_radius_x( ) const { return _max_pixel_radius_x; };

    inline int max_pixel_radius_y( ) const { return _max_pixel_radius_y; };

    inline int min_pixel_radius_x_y( ) const { return _min_pixel_radius_x_y; };

    inline int intra_loop_inc( ) const { return _intra_loop_inc; };

    inline int batch_size( ) const { return _batch_size; };

    inline int n_in_last_batch( ) const { return _n_in_last_batch; };

    inline float GetInPlaneAngle(int z_index_in_batch) const { return _in_plane_angle[index * _batch_size + z_index_in_batch]; };

    inline float GetMirroredOrNot(int z_index_in_batch) const { return _is_search_result_mirrored[index * _batch_size + z_index_in_batch]; };

    IntegerPeak* _peak_buffer;
    IntegerPeak* _d_peak_buffer;

    inline void PrintMemberVariables( ) {
        std::cerr << "BatchedSearch::_n_batches: " << _n_batches << std::endl;
        std::cerr << "BatchedSearch::_batch_size: " << _batch_size << std::endl;
        std::cerr << "BatchedSearch::_n_in_last_batch: " << _n_in_last_batch << std::endl;
        std::cerr << "BatchedSearch::_stride: " << _stride << std::endl;
        std::cerr << "BatchedSearch::_max_pixel_radius_x: " << _max_pixel_radius_x << std::endl;
        std::cerr << "BatchedSearch::_max_pixel_radius_y: " << _max_pixel_radius_y << std::endl;
        std::cerr << "BatchedSearch::_peak_buffer: " << _peak_buffer << std::endl;
        std::cerr << "BatchedSearch::_d_peak_buffer: " << _d_peak_buffer << std::endl;
    }

    // IntegerPeak* _peak_buffer;
    // IntegerPeak* _device_peak_buffer;

    void print_angle_and_mirror( ) {
        for ( int i = 0; i < _in_plane_angle.size( ); i++ ) {
            std::cerr << i << "  angle: " << _in_plane_angle[i] << " mirrored: " << _is_search_result_mirrored[i] << std::endl;
        }
    }

    inline void add_angle_and_mirror(float angle, bool mirror) {
        _in_plane_angle.push_back(angle);
        _is_search_result_mirrored.push_back(mirror);
    }

    inline void add_angle_and_mirror(bool mirror) {
        _in_plane_angle.push_back(_in_plane_angle.back( ));
        _is_search_result_mirrored.push_back(mirror);
    }

  private:
    int _n_search_images;
    int _batch_size;
    int _n_batches;
    int _n_in_last_batch;
    int _intra_loop_inc;
    int _intra_loop_batch_size;

    // min defaults to 0
    int _min_pixel_radius_x_y = {0};
    int _max_pixel_radius_x;
    int _max_pixel_radius_y;
    int _stride;

    bool _test_mirror;

    bool _is_initialized;

    std::vector<bool>  _is_search_result_mirrored;
    std::vector<float> _in_plane_angle;
};
#endif
