if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "`CUDNN_INCLUDE_DIR` is not set.")
endif()
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "`CUDNN_LIBRARY` is not set.")
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE PPLNN_MODEL_ONNX_TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE PPLNN_MODEL_PMX_TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/models/pmx/*.cc)
endif()

set(PPLNN_MODEL_TEST_SRC
    ${PPLNN_MODEL_ONNX_TEST_SRC}
    ${PPLNN_MODEL_PMX_TEST_SRC})

file(GLOB PPLNN_TEST_ENGINE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/*.cc)

file(GLOB_RECURSE PPLNN_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${PPLNN_TEST_ENGINE_SRC}
    ${PPLNN_MODEL_TEST_SRC})

add_executable(pplnn_unittest ${PPLNN_TEST_SRC})

if(PPLNN_ENABLE_ONNX_MODEL)
    if(PPLNN_ONNX_GENERATED_LIBS)
        target_link_libraries(pplnn_unittest PRIVATE ${PPLNN_ONNX_GENERATED_LIBS})
    else()
        target_include_directories(pplnn_unittest PRIVATE ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated)
    endif()
endif()

hpcc_populate_dep(googletest)
target_link_libraries(pplnn_unittest PRIVATE pplnn_static gtest_main cudnn)
target_link_directories(pplnn_unittest PRIVATE ${CUDNN_LIBRARY})

target_include_directories(pplnn_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CUDNN_INCLUDE_DIR}
    ${googletest_SOURCE_DIR}/include)

if(PPLNN_ENABLE_PMX_MODEL)
    target_include_directories(pplnn_unittest PRIVATE
        ${flatbuffers_SOURCE_DIR}/include)
endif()

target_compile_definitions(pplnn_unittest PRIVATE
    PPLNN_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata"
    PPLNN_TESTS_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
