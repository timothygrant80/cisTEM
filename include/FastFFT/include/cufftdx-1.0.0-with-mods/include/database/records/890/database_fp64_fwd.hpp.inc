//Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    32,     0,   404>
    >;
};

template<> struct block_fft_record<14, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    32,     0,   405>
    >;
};

template<> struct block_fft_record<15, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    32,     0,   406>
    >;
};

template<> struct block_fft_record<17, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    32,     0,   407>
    >;
};

template<> struct block_fft_record<18, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    32,     0,   408>
    >;
};

template<> struct block_fft_record<19, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    32,     0,   409>
    >;
};

template<> struct block_fft_record<20, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    32,     0,   410>
    >;
};

template<> struct block_fft_record<21, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    32,     0,   411>
    >;
};

template<> struct block_fft_record<22, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    32,     0,   412>
    >;
};

template<> struct block_fft_record<23, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    32,     0,   413>
    >;
};

template<> struct block_fft_record<24, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    32,     0,   414>
    >;
};

template<> struct block_fft_record<26, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    32,     0,   415>
    >;
};

template<> struct block_fft_record<28, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    32,     0,   416>
    >;
};

template<> struct block_fft_record<29, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    32,     0,   417>
    >;
};

template<> struct block_fft_record<30, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    32,     0,   418>
    >;
};

template<> struct block_fft_record<31, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    32,     0,   419>
    >;
};

template<> struct block_fft_record<2, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,    32,     0,   420>
    >;
};

template<> struct block_fft_record<4, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,    32,     0,   421>,
        block_fft_implementation< 2,    2,    2,    16,    64,   423>,
        block_fft_implementation< 2,    2,    2,    16,    32,   422>
    >;
};

template<> struct block_fft_record<8, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    32,     0,   424>,
        block_fft_implementation< 4,    4,    2,    16,    64,   425>,
        block_fft_implementation< 4,    4,    2,    16,   128,   427>,
        block_fft_implementation< 2,    2,    4,     8,    64,   426>,
        block_fft_implementation< 2,    2,    4,     8,   128,   428>
    >;
};

template<> struct block_fft_record<16, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,    32,     0,   429>,
        block_fft_implementation< 4,    4,    4,     8,   128,   431>,
        block_fft_implementation< 4,    4,    4,     8,   256,   430>,
        block_fft_implementation< 2,    2,    8,     4,   128,   435>,
        block_fft_implementation< 2,    2,    8,     8,   256,   432>,
        block_fft_implementation< 8,    8,    2,    16,   128,   433>,
        block_fft_implementation< 8,    8,    2,    16,   256,   434>
    >;
};

template<> struct block_fft_record<32, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    4,     8,   256,   436>,
        block_fft_implementation< 8,    8,    4,     8,   512,   438>,
        block_fft_implementation< 4,    4,    8,     4,   256,   439>,
        block_fft_implementation< 4,    4,    8,     8,   512,   437>,
        block_fft_implementation< 2,    2,   16,     2,   256,   442>,
        block_fft_implementation< 2,    2,   16,     4,   512,   441>,
        block_fft_implementation<16,   16,    2,    16,   256,   443>,
        block_fft_implementation<16,   16,    2,    16,   512,   440>
    >;
};

template<> struct block_fft_record<64, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,     4,   512,   444>,
        block_fft_implementation< 8,    8,    8,     4,  1024,   447>,
        block_fft_implementation< 4,    4,   16,     4,  1024,   446>,
        block_fft_implementation< 4,    4,   16,     2,   512,   445>,
        block_fft_implementation<16,   16,    4,     8,   512,   449>,
        block_fft_implementation<16,   16,    4,     8,  1024,   448>,
        block_fft_implementation< 2,    2,   32,     1,   512,   451>,
        block_fft_implementation< 2,    2,   32,     2,  1024,   450>
    >;
};

template<> struct block_fft_record<128, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    8,     4,  1024,   454>,
        block_fft_implementation<16,   16,    8,     4,  2048,   453>,
        block_fft_implementation< 4,    4,   32,     2,  2048,   455>,
        block_fft_implementation< 4,    4,   32,     1,  1024,   456>,
        block_fft_implementation< 2,    2,   64,     1,  1024,   459>,
        block_fft_implementation< 2,    2,   64,     1,  2048,   458>,
        block_fft_implementation< 8,    8,   16,     2,  1024,   452>,
        block_fft_implementation< 8,    8,   16,     4,  2048,   457>
    >;
};

template<> struct block_fft_record<256, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,     2,  4096,   461>,
        block_fft_implementation< 4,    4,   64,     1,  2048,   464>,
        block_fft_implementation<16,   16,   16,     2,  2048,   462>,
        block_fft_implementation< 4,    4,   64,     1,  4096,   463>,
        block_fft_implementation< 8,    8,   32,     2,  4096,   465>,
        block_fft_implementation< 8,    8,   32,     1,  2048,   460>,
        block_fft_implementation< 2,    2,  128,     1,  2048,   466>,
        block_fft_implementation< 2,    2,  128,     1,  4096,   467>
    >;
};

template<> struct block_fft_record<512, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     1,  4096,   468>,
        block_fft_implementation< 8,    8,   64,     1,  8192,   469>,
        block_fft_implementation< 4,    4,  128,     1,  4096,   472>,
        block_fft_implementation< 4,    4,  128,     1,  8192,   473>,
        block_fft_implementation< 2,    2,  256,     1,  8192,   475>,
        block_fft_implementation< 2,    2,  256,     1,  4096,   474>,
        block_fft_implementation<16,   16,   32,     1,  4096,   471>,
        block_fft_implementation<16,   16,   32,     1,  8192,   470>
    >;
};

template<> struct block_fft_record<1024, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,  256,     1,  8192,   478>,
        block_fft_implementation< 4,    4,  256,     1, 16384,   479>,
        block_fft_implementation<16,   16,   64,     1,  8192,   477>,
        block_fft_implementation< 2,    2,  512,     1, 16384,   482>,
        block_fft_implementation<16,   16,   64,     1, 16384,   480>,
        block_fft_implementation< 2,    2,  512,     1,  8192,   483>,
        block_fft_implementation< 8,    8,  128,     1,  8192,   476>,
        block_fft_implementation< 8,    8,  128,     1, 16384,   481>
    >;
};

template<> struct block_fft_record<2048, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  128,     1, 16384,   485>,
        block_fft_implementation<16,   16,  128,     1, 32768,   486>,
        block_fft_implementation< 4,    4,  512,     1, 32768,   489>,
        block_fft_implementation< 4,    4,  512,     1, 16384,   488>,
        block_fft_implementation< 8,    8,  256,     1, 32768,   487>,
        block_fft_implementation< 8,    8,  256,     1, 16384,   484>,
        block_fft_implementation< 2,    2, 1024,     1, 32768,   491>,
        block_fft_implementation< 2,    2, 1024,     1, 16384,   490>
    >;
};

template<> struct block_fft_record<4096, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4, 1024,     1, 65536,   497>,
        block_fft_implementation< 4,    4, 1024,     1, 32768,   496>,
        block_fft_implementation< 8,    8,  512,     1, 65536,   495>,
        block_fft_implementation<16,   16,  256,     1, 32768,   493>,
        block_fft_implementation< 8,    8,  512,     1, 32768,   492>,
        block_fft_implementation<16,   16,  256,     1, 65536,   494>
    >;
};

template<> struct block_fft_record<8192, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8, 1024,     1, 65536,   499>,
        block_fft_implementation<16,   16,  512,     1, 65536,   498>
    >;
};

template<> struct block_fft_record<3, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,    32,     0,   500>
    >;
};

template<> struct block_fft_record<9, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,    32,     0,   501>,
        block_fft_implementation< 3,    3,    3,    32,   144,   503>,
        block_fft_implementation< 3,    3,    3,    32,    72,   502>
    >;
};

template<> struct block_fft_record<27, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,    32,     0,   504>,
        block_fft_implementation< 3,    3,    9,    32,   432,   508>,
        block_fft_implementation< 3,    3,    9,     7,   216,   507>,
        block_fft_implementation< 9,    9,    3,    32,   432,   506>,
        block_fft_implementation< 9,    9,    3,    32,   216,   505>
    >;
};

template<> struct block_fft_record<81, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,   27,    13,  1296,   513>,
        block_fft_implementation< 3,    3,   27,    13,   648,   514>,
        block_fft_implementation< 9,    9,    9,    32,  1296,   509>,
        block_fft_implementation< 9,    9,    9,    32,   648,   510>,
        block_fft_implementation<27,   27,    3,    32,  1296,   512>,
        block_fft_implementation<27,   27,    3,    32,   648,   511>
    >;
};

template<> struct block_fft_record<243, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,   81,     9,  3888,   519>,
        block_fft_implementation<27,   27,    9,     7,  1944,   515>,
        block_fft_implementation<27,   27,    9,     7,  3888,   516>,
        block_fft_implementation< 3,    3,   81,     9,  1944,   520>,
        block_fft_implementation< 9,    9,   27,    13,  3888,   517>,
        block_fft_implementation< 9,    9,   27,    13,  1944,   518>
    >;
};

template<> struct block_fft_record<729, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,  243,     3, 11664,   526>,
        block_fft_implementation< 3,    3,  243,     3,  5832,   525>,
        block_fft_implementation<27,   27,   27,    13,  5832,   523>,
        block_fft_implementation<27,   27,   27,     7, 11664,   524>,
        block_fft_implementation< 9,    9,   81,     7, 11664,   521>,
        block_fft_implementation< 9,    9,   81,     9,  5832,   522>
    >;
};

template<> struct block_fft_record<2187, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,  729,     1, 34992,   530>,
        block_fft_implementation< 3,    3,  729,     1, 17496,   529>,
        block_fft_implementation< 9,    9,  243,     3, 17496,   527>,
        block_fft_implementation< 9,    9,  243,     1, 34992,   528>
    >;
};

template<> struct block_fft_record<6561, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  729,     1, 52488,   531>
    >;
};

template<> struct block_fft_record<5, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    32,     0,   532>
    >;
};

template<> struct block_fft_record<25, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,    32,     0,   533>,
        block_fft_implementation< 5,    5,    5,    32,   400,   534>,
        block_fft_implementation< 5,    5,    5,    32,   200,   535>
    >;
};

template<> struct block_fft_record<125, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,   25,    14,  2000,   536>,
        block_fft_implementation< 5,    5,   25,    14,  1000,   537>,
        block_fft_implementation<25,   25,    5,    32,  1000,   538>,
        block_fft_implementation<25,   25,    5,    32,  2000,   539>
    >;
};

template<> struct block_fft_record<625, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,    14,  5000,   540>,
        block_fft_implementation<25,   25,   25,     5, 10000,   543>,
        block_fft_implementation< 5,    5,  125,     1,  5000,   541>,
        block_fft_implementation< 5,    5,  125,     1, 10000,   542>
    >;
};

template<> struct block_fft_record<3125, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,  625,     1, 50000,   544>,
        block_fft_implementation< 5,    5,  625,     1, 25000,   545>
    >;
};

template<> struct block_fft_record<7, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,    32,     0,   546>
    >;
};

template<> struct block_fft_record<49, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    32,   392,   547>,
        block_fft_implementation< 7,    7,    7,    32,   784,   548>
    >;
};

template<> struct block_fft_record<343, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,    13,  5488,   549>,
        block_fft_implementation< 7,    7,   49,    13,  2744,   550>
    >;
};

template<> struct block_fft_record<2401, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     1, 38416,   551>,
        block_fft_implementation< 7,    7,  343,     1, 19208,   552>
    >;
};

template<> struct block_fft_record<11, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,    32,     0,   553>
    >;
};

template<> struct block_fft_record<121, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    29,   968,   555>,
        block_fft_implementation<11,   11,   11,    29,  1936,   554>
    >;
};

template<> struct block_fft_record<1331, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     5, 10648,   556>,
        block_fft_implementation<11,   11,  121,     1, 21296,   557>
    >;
};

template<> struct block_fft_record<6, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    32,     0,   558>
    >;
};

template<> struct block_fft_record<36, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    16,   576,   559>,
        block_fft_implementation< 6,    6,    6,    16,   288,   560>
    >;
};

template<> struct block_fft_record<216, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,     8,  3456,   562>,
        block_fft_implementation< 6,    6,   36,     8,  1728,   561>
    >;
};

template<> struct block_fft_record<1296, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     4, 20736,   564>,
        block_fft_implementation< 6,    6,  216,     4, 10368,   563>
    >;
};

template<> struct block_fft_record<10, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    32,     0,   565>
    >;
};

template<> struct block_fft_record<100, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    16,  1600,   566>,
        block_fft_implementation<10,   10,   10,    16,   800,   567>
    >;
};

template<> struct block_fft_record<1000, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,     8,  8000,   569>,
        block_fft_implementation<10,   10,  100,     6, 16000,   568>
    >;
};

template<> struct block_fft_record<12, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,    32,     0,   570>
    >;
};

template<> struct block_fft_record<144, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,     8,  1152,   571>,
        block_fft_implementation<12,   12,   12,     8,  2304,   572>
    >;
};

template<> struct block_fft_record<1728, double, fft_type::c2c, fft_direction::forward, 890> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     2, 27648,   574>,
        block_fft_implementation<12,   12,  144,     2, 13824,   573>
    >;
};
