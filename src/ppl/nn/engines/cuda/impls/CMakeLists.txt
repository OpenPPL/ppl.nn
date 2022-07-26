cmake_minimum_required(VERSION 3.14)
project(PPLCUDAKernel)

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "cannot find python3")
endif()

# ---------------------------------------------------------------------------------------- #

set(__CUDA_CONV_SRC__)

if(PPLNN_ENABLE_CUDA_JIT)
    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/gene_header.py ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv ${CMAKE_CURRENT_BINARY_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(__gene_kernel_src__ ${CMAKE_CURRENT_BINARY_DIR}/gene_header.cc src/nn/conv/gene_kernel.cc)
    if(CMAKE_COMPILER_IS_GNUCC)
        set_source_files_properties(${__gene_kernel_src__} PROPERTIES COMPILE_FLAGS -Werror=non-virtual-dtor)
    endif()
    list(APPEND __CUDA_CONV_SRC__ ${__gene_kernel_src__})
    unset(__gene_kernel_src__)

else()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
        execute_process(
            # sm75 kernels
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm75_fp16_hmma1688_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm75/fp16/hmma1688
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_sm75_fp16_hmma1688_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm75/fp16/hmma1688
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm75_fp16_hmma1688_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm75/fp16/hmma1688

            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm75_int8_imma8816_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm75/int8/imma8816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_sm75_int8_imma8816_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm75/int8/imma8816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm75_int8_imma8816_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm75/int8/imma8816

            OUTPUT_STRIP_TRAILING_WHITESPACE)

        file(GLOB_RECURSE __SM75_CONV_SRC__
            ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm75/*/*/idxn_kernels.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm75/*/*/f*_kernels.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm75/*/*/f*_kernels*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm75/*/*/init*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm75/*/*/init*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm75/*/*/init*.cu)

        if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
            if(CUDA_VERSION VERSION_GREATER_EQUAL "11.7")
                set_source_files_properties(${__SM75_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                    "-gencode arch=compute_87,code=sm_87 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 -fno-var-tracking-assignments")
            elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
                set_source_files_properties(${__SM75_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                    "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 -fno-var-tracking-assignments")
            else()
                set_source_files_properties(${__SM75_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                    "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 -fno-var-tracking-assignments")
            endif()
        else()
            set_source_files_properties(${__SM75_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                "-gencode arch=compute_75,code=sm_75 -fno-var-tracking-assignments")
        endif()
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
        # sm80 kernels
        # fp16
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm80_fp16_hmma1688_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/fp16/hmma1688
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm80_fp16_hmma1688_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/fp16/hmma1688

            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm80_fp16_hmma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/fp16/hmma16816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_sm80_fp16_hmma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm80/fp16/hmma16816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm80_fp16_hmma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/fp16/hmma16816

            # int8
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm80_int8_imma8816_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/int8/imma8816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm80_int8_imma8816_kernel.py  ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/int8/imma8816

            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm80_int8_imma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/int8/imma16816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_sm80_int8_imma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm80/int8/imma16816
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm80_int8_imma16816_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/int8/imma16816

            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_sm80_int8_imma16832_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/int8/imma16832
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_sm80_int8_imma16832_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm80/int8/imma16832
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_sm80_int8_imma16832_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/int8/imma16832

            OUTPUT_STRIP_TRAILING_WHITESPACE)

        file(GLOB_RECURSE __SM80_CONV_SRC__
            ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm80/*/*/idxn_kernels.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/*/*/f*_kernels.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/*/*/f*_kernels*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/sm80/*/*/init*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/sm80/*/*/init*.cu
            ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/sm80/*/*/init*.cu)

        if(CUDA_VERSION VERSION_GREATER_EQUAL "11.7")
            set_source_files_properties(${__SM80_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                "-gencode arch=compute_87,code=sm_87 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -fno-var-tracking-assignments")
        elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
            set_source_files_properties(${__SM80_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -fno-var-tracking-assignments")
        else()
            set_source_files_properties(${__SM80_CONV_SRC__} PROPERTIES COMPILE_FLAGS
                "-gencode arch=compute_80,code=sm_80 -fno-var-tracking-assignments")
        endif()
    endif()

    list(APPEND __CUDA_CONV_SRC__ ${__SM80_CONV_SRC__} ${__SM75_CONV_SRC__})
    unset(__SM80_CONV_SRC__)
    unset(__SM75_CONV_SRC__)
endif()

list(APPEND __CUDA_SRC__ ${__CUDA_CONV_SRC__})
unset(__CUDA_CONV_SRC__)

# --------------------------------------------------------------------------- #

# cuda tools
#list(APPEND __CUDA_SRC__ tools/*.cu)

file(GLOB_RECURSE __CUDA_OTHERS_SRC__
    src/arithmetic/*.cu
    src/memory/*.cu
    src/reduce/*.cu
    src/reformat/*.cu
    src/nn/conv/conv_*.cu
    src/nn/conv/common/*.cu
    src/unary/*.cu)

file(GLOB __TMP__ src/nn/*.cu)
list(APPEND __CUDA_OTHERS_SRC__ ${__TMP__})
file(GLOB __TMP__ src/nn/depthwise/*.cu)
list(APPEND __CUDA_OTHERS_SRC__ ${__TMP__})
unset(__TMP__)

set(_NVCC_FLAGS )
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "9")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "11")
    if(CUDA_VERSION_MINOR VERSION_GREATER_EQUAL "4")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80")
    elseif(CUDA_VERSION_MINOR VERSION_GREATER_EQUAL "1")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80")
    else()
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    endif()
endif()
set_source_files_properties(${__CUDA_OTHERS_SRC__} PROPERTIES COMPILE_FLAGS ${_NVCC_FLAGS})
unset(_NVCC_FLAGS)

list(APPEND __CUDA_SRC__ ${__CUDA_OTHERS_SRC__})
unset(__CUDA_OTHERS_SRC__)

# ----------------------------------------------------------------------------------------------- #

if(PPLNN_ENABLE_CUDA_JIT)
    list(APPEND __CUDA_SRC__ src/nn/conv/conv_jit.cc)
endif()
list(APPEND __CUDA_SRC__ src/nn/conv/common/init_lut.cc)

# ----------------------------------------------------------------------------------------------- #

add_library(pplkernelcuda_static STATIC ${__CUDA_SRC__} ${PPLNN_CUDA_EXTERNAL_SOURCES})
target_compile_features(pplkernelcuda_static PRIVATE cxx_std_11)

get_filename_component(__PPLNN_SOURCE_DIR__ "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${__PPLNN_SOURCE_DIR__}/include ${__PPLNN_SOURCE_DIR__}/src)
unset(__PPLNN_SOURCE_DIR__)

target_include_directories(pplkernelcuda_static PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv
    ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES}
    ${PPLNN_CUDA_EXTERNAL_INCLUDE_DIRECTORIES})
target_link_directories(pplkernelcuda_static PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES} ${PPLNN_CUDA_EXTERNAL_LINK_DIRECTORIES})

# `PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES` is needed for generating pplkernecuda-config.cmake
set(PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES cuda cudart)
if(PPLNN_ENABLE_CUDA_JIT)
    list(APPEND PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES nvrtc)
endif()

target_link_libraries(pplkernelcuda_static PUBLIC ${PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES} ${PPLNN_CUDA_EXTERNAL_LINK_LIBRARIES})

hpcc_populate_dep(pplcommon)
target_link_libraries(pplkernelcuda_static PUBLIC pplcommon_static)

if(PPLNN_ENABLE_CUDA_JIT)
    target_compile_definitions(pplkernelcuda_static PUBLIC PPLNN_ENABLE_CUDA_JIT)
endif()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelcuda_static DESTINATION lib)

    set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplkernelcuda-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pplkernelcuda-config.cmake.in
        ${__PPLNN_CMAKE_CONFIG_FILE__}
        @ONLY)
    install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
    unset(__PPLNN_CMAKE_CONFIG_FILE__)
endif()

unset(PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES)
unset(__CUDA_SRC__)
