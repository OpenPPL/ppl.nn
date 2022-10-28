file(GLOB PPLNN_TOOL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)

add_executable(pplnn ${PPLNN_TOOL_SRC})
target_link_libraries(pplnn PRIVATE pplnn_basic_static)
target_compile_features(pplnn PRIVATE cxx_std_11)

if(PPLNN_USE_X86)
   target_link_libraries(pplnn PRIVATE pplnn_x86_static)
endif()

if(PPLNN_USE_CUDA)
   target_link_libraries(pplnn PRIVATE pplnn_cuda_static)
endif()

if(PPLNN_USE_ARM)
   target_link_libraries(pplnn PRIVATE pplnn_arm_static)
endif()

if(PPLNN_USE_RISCV)
   target_link_libraries(pplnn PRIVATE pplnn_riscv_static)
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    target_link_libraries(pplnn PRIVATE pplnn_onnx_static)
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    target_link_libraries(pplnn PRIVATE pplnn_pmx_static)
endif()
