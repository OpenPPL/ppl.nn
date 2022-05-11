if (PPLNN_USE_AARCH64 AND PPLNN_USE_NUMA)
    add_executable(pplnn_multi_numa_node pplnn_multi_numa_node.cc
                                         ${PROJECT_SOURCE_DIR}/tools/simple_flags.cc)
    target_compile_features(pplnn_multi_numa_node PRIVATE cxx_std_11)
    target_include_directories(pplnn_multi_numa_node PRIVATE ${PROJECT_SOURCE_DIR}/tools)
    target_link_libraries(pplnn_multi_numa_node PUBLIC pplnn_static)
endif()
