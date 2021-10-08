file(GLOB_RECURSE PPLNN_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(pplnn_unittest ${PPLNN_TEST_SRC})

hpcc_populate_dep(googletest)
target_link_libraries(pplnn_unittest PRIVATE pplnn_static gtest_main)

target_include_directories(pplnn_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${googletest_SOURCE_DIR}/include)

target_compile_definitions(pplnn_unittest PRIVATE PPLNN_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
target_compile_features(pplnn_unittest PRIVATE cxx_std_11)
