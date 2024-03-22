set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplnnllm-config.cmake)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pplnnllm-config.cmake.in
    ${__PPLNN_CMAKE_CONFIG_FILE__}
    @ONLY)
install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
unset(__PPLNN_CMAKE_CONFIG_FILE__)

install(TARGETS ${PPLNN_ONNX_GENERATED_LIBS} DESTINATION lib)
