file(GLOB PPLNN_TOOL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)

add_executable(pplnn ${PPLNN_TOOL_SRC})
target_link_libraries(pplnn PRIVATE pplnn_static)
target_compile_features(pplnn PRIVATE cxx_std_11)
