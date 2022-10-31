hpcc_populate_dep(luacpp)

# --------------------------------------------------------------------------- #

file(GLOB __PPLNN_LUA_API_SRC__
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lua_pplnn.cc)

if(PPLNN_USE_X86)
    file(GLOB_RECURSE __PPLNN_LUA_X86_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/x86/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_X86_SRC__})
    unset(__PPLNN_LUA_X86_SRC__)
endif()

if(PPLNN_USE_CUDA)
    file(GLOB_RECURSE __PPLNN_LUA_CUDA_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/cuda/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_CUDA_SRC__})
    unset(__PPLNN_LUA_CUDA_SRC__)
endif()

if(PPLNN_USE_RISCV)
    file(GLOB_RECURSE __PPLNN_LUA_RISCV_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/riscv/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_RISCV_SRC__})
    unset(__PPLNN_LUA_RISCV_SRC__)
endif()

if(PPLNN_USE_ARM)
    file(GLOB_RECURSE __PPLNN_LUA_ARM_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/arm/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_ARM_SRC__})
    unset(__PPLNN_LUA_ARM_SRC__)
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE __PPLNN_LUA_ONNX_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_ONNX_SRC__})
    unset(__PPLNN_LUA_ONNX_SRC__)
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE __PPLNN_LUA_PMX_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/models/pmx/*.cc)
    list(APPEND __PPLNN_LUA_API_SRC__ ${__PPLNN_LUA_PMX_SRC__})
    unset(__PPLNN_LUA_PMX_SRC__)
endif()

if(NOT PPLNN_SOURCE_EXTERNAL_LUA_SOURCES)
    set(PPLNN_SOURCE_EXTERNAL_LUA_SOURCES lua_dummy_load_resources.cc)
endif()

add_library(luapplnn_shared SHARED ${__PPLNN_LUA_API_SRC__} ${PPLNN_SOURCE_EXTERNAL_LUA_SOURCES})
unset(__PPLNN_LUA_API_SRC__)

target_link_libraries(luapplnn_shared PRIVATE
    pplnn_static luacpp_static
    ${PPLNN_SOURCE_EXTERNAL_LUA_LINK_LIBRARIES})
target_include_directories(luapplnn_shared PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PPLNN_SOURCE_EXTERNAL_LUA_INCLUDE_DIRECTORIES})

# --------------------------------------------------------------------------- #

# rename target for lua import
set_target_properties(luapplnn_shared PROPERTIES PREFIX "")
set_target_properties(luapplnn_shared PROPERTIES OUTPUT_NAME "nn")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(luapplnn_shared PROPERTIES SUFFIX ".so") # using .so instead of .dylib
endif()

if(PPLNN_INSTALL)
    install(TARGETS luapplnn_shared DESTINATION lib/luappl)
endif()
