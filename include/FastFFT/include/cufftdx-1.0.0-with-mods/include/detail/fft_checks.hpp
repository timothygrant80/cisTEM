// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_CHECKS_HPP
#define CUFFTDX_DETAIL_FFT_CHECKS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../operators.hpp"
#include "../traits/detail/bluestein_helpers.hpp"
#include "../traits/fft_traits.hpp"

namespace cufftdx {
    namespace detail {

// SM70
#define CUFFTDX_DETAIL_SM700_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP64_MAX 8192
// SM72
#define CUFFTDX_DETAIL_SM720_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP64_MAX 8192
// SM75
#define CUFFTDX_DETAIL_SM750_FP16_MAX 4096
#define CUFFTDX_DETAIL_SM750_FP32_MAX 4096
#define CUFFTDX_DETAIL_SM750_FP64_MAX 2048
// SM80
#define CUFFTDX_DETAIL_SM800_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP64_MAX 16384
// SM86
#define CUFFTDX_DETAIL_SM860_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP64_MAX 8192
// SM87
#define CUFFTDX_DETAIL_SM870_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM870_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM870_FP64_MAX 16384
// SM89
#define CUFFTDX_DETAIL_SM890_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM890_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM890_FP64_MAX 8192
// SM90
#define CUFFTDX_DETAIL_SM900_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM900_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM900_FP64_MAX 16384

// Thread FFT
#define CUFFTDX_DETAIL_THREAD_FP16 32
#define CUFFTDX_DETAIL_THREAD_FP32 32
#define CUFFTDX_DETAIL_THREAD_FP64 16

        template<class Precision, unsigned int Size>
        class is_supported_thread
        {
        public:
            static constexpr bool fp16_thread_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_THREAD_FP16) && (Size >= 2));
            static constexpr bool fp32_thread_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_THREAD_FP32) && (Size >= 2));
            static constexpr bool fp64_thread_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_THREAD_FP64) && (Size >= 2));

            static constexpr bool thread_value = fp16_thread_value || fp32_thread_value || fp64_thread_value;
        };

        template<class Precision, unsigned int Size, unsigned int Arch>
        class is_supported
        {
        public:
            static constexpr bool thread_value     = false;
            static constexpr bool block_value      = false;
            static constexpr bool blue_block_value = false;
            static constexpr bool value            = false;
        };

        // Max supported sizes, ignores SM
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, unsigned(-1)>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM900_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM70
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 700>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM700_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM700_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM72
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 720>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM720_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM720_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM75
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 750>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM750_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM750_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM80
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 800>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM800_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM86
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 860>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM860_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM860_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM87
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 870>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM870_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM870_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM870_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM870_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM89
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 890>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM890_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM890_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM890_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM890_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        // SM90
        template<class Precision, unsigned int Size>
        class is_supported<Precision, Size, 900>: public is_supported_thread<Precision, Size>
        {
            static constexpr auto blue_size = detail::get_bluestein_size(Size);

        public:
            static constexpr bool fp16_block_value =
                CUFFTDX_STD::is_same<__half, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP16_MAX) && (Size >= 2));
            static constexpr bool fp32_block_value =
                CUFFTDX_STD::is_same<float, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP32_MAX) && (Size >= 2));
            static constexpr bool fp64_block_value =
                CUFFTDX_STD::is_same<double, Precision>::value && ((Size <= CUFFTDX_DETAIL_SM900_FP64_MAX) && (Size >= 2));

            static constexpr bool block_value = fp16_block_value || fp32_block_value || fp64_block_value;
            static constexpr bool blue_block_value = ((blue_size <= CUFFTDX_DETAIL_SM900_FP64_MAX) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        template<class Precision,
                 class EPT,
                 class block_fft_record_t,
                 bool PresentInDatabase = block_fft_record_t::defined>
        struct is_ept_supported: public CUFFTDX_STD::false_type {};

        template<class Precision, class EPT, class block_fft_record_t>
        struct is_ept_supported<Precision, EPT, block_fft_record_t, true> {
            // Get default implementation
            using default_block_config_t =
                typename database::detail::type_list_element<0, typename block_fft_record_t::blobs>::type;
            // Get default EPT
            using default_ept = ElementsPerThread<default_block_config_t::elements_per_thread>;
            // Select transposition types to look for in the database
            #ifdef CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_X_TRANSPOSITION
            static constexpr unsigned int this_fft_trp_option_v = 1;
            #elif defined(CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_XY_TRANSPOSITION)
            static constexpr unsigned int this_fft_trp_option_v = 2;
            #else
            static constexpr unsigned int this_fft_trp_option_v = 0;
            #endif
            // Search for implementation
            using this_fft_elements_per_thread =
                CUFFTDX_STD::conditional_t<!CUFFTDX_STD::is_void<EPT>::value, EPT, default_ept>;
            static constexpr auto this_fft_ept_v = this_fft_elements_per_thread::value;
            using this_fft_block_fft_implementation =
                typename database::detail::search_by_ept<this_fft_ept_v,
                                                         Precision,
                                                         this_fft_trp_option_v,
                                                         typename block_fft_record_t::blobs>::type;
            static constexpr bool implementation_exists =
                !CUFFTDX_STD::is_void<this_fft_block_fft_implementation>::value;

        public:
            static constexpr bool value = CUFFTDX_STD::is_void<EPT>::value ? true : implementation_exists;
        };

        template<class Precision,
                 fft_type      Type,
                 fft_direction Direction,
                 unsigned      Size,
                 bool          Block,
                 bool          Thread,
                 class EPT, // void if not set
                 unsigned int Arch>
        class is_supported_helper
        {
            // Checks
            static_assert(Block || Thread,
                          "To check if an FFT description is supported on a given architecture it has to have Block or "
                          "Thead execution operator");

            static constexpr bool is_supported_thread     = is_supported<Precision, Size, Arch>::thread_value;
            static constexpr bool is_supported_block      = is_supported<Precision, Size, Arch>::block_value;
            static constexpr bool is_supported_block_blue = is_supported<Precision, Size, Arch>::blue_block_value;

            static constexpr bool requires_block_blue =
                Block && is_bluestein_required<Size, Precision, Direction, Type, Arch>::value;

            // Check if EPT is supported
            using block_fft_record_t =
                cufftdx::database::detail::block_fft_record<Size, Precision, Type, Direction, Arch>;
            static constexpr bool is_ept_supported_v = is_ept_supported<Precision, EPT, block_fft_record_t>::value;

            // Check if EPT is supported
            static constexpr auto blue_size = detail::get_bluestein_size(Size);
            using block_fft_record_blue_t =
                cufftdx::database::detail::block_fft_record<blue_size, Precision, Type, Direction, Arch>;
            static constexpr bool is_ept_supported_blue_v = is_ept_supported<Precision, EPT, block_fft_record_blue_t>::value;

        public:
            static constexpr bool value =
                (Thread && is_supported_thread) ||                                               // Thread
                (Block && is_supported_block && is_ept_supported_v) ||                           // Block
                (Block && is_supported_block_blue && requires_block_blue && is_ept_supported_blue_v); // Blue
        };
    } // namespace detail

    // Check if description is supported on given Architecture
    template<class Description, unsigned int Architecture>
    struct is_supported:
        public CUFFTDX_STD::bool_constant<
            detail::is_supported_helper<precision_of_t<Description>,
                                        type_of<Description>::value,
                                        direction_of<Description>::value,
                                        size_of<Description>::value,
                                        detail::has_operator<fft_operator::block, Description>::value,
                                        detail::has_operator<fft_operator::thread, Description>::value,
                                        detail::get_t<fft_operator::elements_per_thread, Description>,
                                        Architecture>::value> {};

    template<class Description, unsigned int Architecture>
    inline constexpr bool is_supported_v = is_supported<Description, Architecture>::value;
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_FFT_CHECKS_HPP
