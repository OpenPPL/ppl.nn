if(NOT LUA_INCLUDE_DIR OR NOT LUA_LIBRARIES)
    find_package(Lua QUIET)
    if (NOT Lua_FOUND)
        message(FATAL_ERROR "lua dev lib is required. please install lua development files or specify `LUA_INCLUDE_DIR` and `LUA_LIBRARIES` manually.")
    endif()
endif()

hpcc_populate_dep(luacpp)

file(GLOB_RECURSE PPLNN_LUA_API_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(luapplnn_shared SHARED ${PPLNN_LUA_API_SRC})
target_compile_features(luapplnn_shared PRIVATE cxx_std_14)
target_link_libraries(luapplnn_shared PUBLIC pplnn_static luacpp_static ${LUA_LIBRARIES})
target_include_directories(luapplnn_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${LUA_INCLUDE_DIR}
    ${HPCC_DEPS_DIR}/luacpp)

# rename target for lua import
set_target_properties(luapplnn_shared PROPERTIES PREFIX "")
set_target_properties(luapplnn_shared PROPERTIES OUTPUT_NAME "nn")

if(PPLNN_INSTALL)
    install(TARGETS luapplnn_shared DESTINATION luappl)
endif()
