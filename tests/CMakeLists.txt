if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE PPLNN_MODEL_ONNX_TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
endif()
set(PPLNN_MODEL_TEST_SRC ${PPLNN_MODEL_ONNX_TEST_SRC})

file(GLOB_RECURSE PPLNN_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/params/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${PPLNN_MODEL_TEST_SRC})

add_executable(pplnn_unittest ${PPLNN_TEST_SRC})

hpcc_populate_dep(googletest)
target_link_libraries(pplnn_unittest PRIVATE pplnn_static gtest_main)

target_include_directories(pplnn_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${flatbuffers_SOURCE_DIR}/include
    ${googletest_SOURCE_DIR}/include)

target_compile_definitions(pplnn_unittest PRIVATE PPLNN_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
target_compile_features(pplnn_unittest PRIVATE cxx_std_11)
