#include <iostream>

constexpr int M = 1;
template < typename T, int N >
struct my_functor {

    if constexpr ( M == 1 ) {
        __device__ __forceinline__
        T operator () (T x, T y) const {
            return x + y;
        }
    } 
    else if constexpr (N==2) {
        static __device__ __forceinline__
        T operator () (T x, T y) const {
            return x * y;
        }
    }
    else{
        static __device__ __forceinline__
        T operator () (T x, T y) const {
            printf("X is %f and Y is % f\n", x, y);
        }
    }

};

}
template <typename T>
struct add {
    __device__
    T operator () (T x, T y) const {
        return x + y;
    }
};

template <typename T>
using add_t = add<T>(T x, T y);

template <typename T>
struct noop {
    __device__
    T operator () () const {
        return T(6);
    }
};


template <typename T, template<T, N> class OP> 
__global__ void kernel(T * d_x, T * d_y, T * result)
{
    printf("in the func kernel\n");
    // *result = (*op)(*d_x, *d_y);
    *result = OP(*d_x, *d_y);
}



template < typename T, typename FunctionType >
auto set_fp() -> FunctionType {

    if constexpr(std::is_same_v<FunctionType, noop<T>>)
    {
        noop_t<T> d_ptr;
        cudaMemcpyFromSymbol(d_ptr, p_noop<T>, sizeof(p_noop<T>));
        std::cout <<  "mul_func0: " << std::endl;
        return d_ptr;
    }
    else 
    if constexpr(std::is_same_v<FunctionType, add<T>>)
    {
        add<T>d_ptr();
        cudaMemcpyFromSymbol(&d_ptr, p_add<T>, sizeof(p_add<T>));
        std::cout << "mul_func: " << std::endl;
        return d_ptr;
    }
    else
    {
        static_assert(std::is_same_v<FunctionType, noop<T>> || std::is_same_v<FunctionType, add<T>>, "FunctionType must be either noop or add");
        std::cout << "Invalid function type" << std::endl;
    }

}
template < typename T,  int N > 
void test(T x, T y) {


    T * d_x, * d_y;
    cudaMalloc(&d_x, sizeof(T));
    cudaMalloc(&d_y, sizeof(T));
    cudaMemcpy(d_x, &x, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y, sizeof(T), cudaMemcpyHostToDevice);

    T result;
    T * d_result, * h_result;
    cudaMalloc(&d_result, sizeof(T));
    h_result = &result;



    kernel<T, functor<T, N> ><<<1, 1>>>d_x, d_y, d_result);
    cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << func_name << result << std::endl;

//     kernel<T><<<1,1>>>(h_add_func, d_x, d_y, d_result);
//     cudaDeviceSynchronize();
//     cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
//     std::cout << "Sum: " << result << std::endl;

//     kernel<T><<<1,1>>>(h_mul_func, d_x, d_y, d_result);
//     cudaDeviceSynchronize();
//     cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
//     std::cout << "Product: " << result << std::endl;
}

int main()
{
    std::cout << "Test int for type int ..." << std::endl;
    // test<>(2, 10);
    static_assert( std::is_same_v<noop<int>, noop<int>>, "noop<int> must be same as noop<int>");

    std::cout << "Test float for type 0 ..." << std::endl;
    test<float, 0>(2.05, 10.00);

    std::cout << "Test float for type 1 ..." << std::endl;
    test<float, 1>(2.05, 10.00);

    std::cout << "Test double for type 2 ..." << std::endl;
    test<double, 2>(2.05, 10.00);
}
