hpcc_populate_dep(luacpp)

# --------------------------------------------------------------------------- #

file(GLOB PPLNN_LUA_API_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lua_pplnn.cc)

if(PPLNN_USE_X86)
    file(GLOB_RECURSE PPLNN_LUA_X86_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/x86/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_X86_SRC})
endif()

if(PPLNN_USE_CUDA)
    file(GLOB_RECURSE PPLNN_LUA_CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/cuda/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_CUDA_SRC})
endif()

if(PPLNN_USE_RISCV)
    file(GLOB_RECURSE PPLNN_LUA_RISCV_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/riscv/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_RISCV_SRC})
endif()

if(PPLNN_USE_ARM)
    file(GLOB_RECURSE PPLNN_LUA_ARM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/arm/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_ARM_SRC})
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE PPLNN_LUA_ONNX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_ONNX_SRC})
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE PPLNN_LUA_PMX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/models/pmx/*.cc)
    list(APPEND PPLNN_LUA_API_SRC ${PPLNN_LUA_PMX_SRC})
endif()

add_library(luapplnn_shared SHARED ${PPLNN_LUA_API_SRC})
target_link_libraries(luapplnn_shared PUBLIC pplnn_static luacpp_static)
target_include_directories(luapplnn_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

# --------------------------------------------------------------------------- #

# rename target for lua import
set_target_properties(luapplnn_shared PROPERTIES PREFIX "")
set_target_properties(luapplnn_shared PROPERTIES OUTPUT_NAME "nn")

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(luapplnn_shared PROPERTIES SUFFIX ".so") # using .so instead of .dylib
endif()

if(PPLNN_INSTALL)
    install(TARGETS luapplnn_shared DESTINATION lib/luappl)
endif()
