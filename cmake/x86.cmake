set(PPLNN_USE_X86 ON)

file(GLOB_RECURSE __PPLNN_X86_SRC__ src/ppl/nn/engines/x86/*.cc)
add_library(pplnn_x86_static STATIC ${PPLNN_SOURCE_EXTERNAL_X86_ENGINE_SOURCES} ${__PPLNN_X86_SRC__})
unset(__PPLNN_X86_SRC__)

add_subdirectory(src/ppl/nn/engines/x86/impls)
target_link_libraries(pplnn_x86_static PUBLIC pplnn_basic_static pplkernelx86_static)

target_compile_definitions(pplnn_x86_static PUBLIC PPLNN_USE_X86)

if(PPLNN_ENABLE_SANITIZE_OPTIONS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(__ASAN_FLAGS__ "-fsanitize=undefined -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${__ASAN_FLAGS__}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${__ASAN_FLAGS__}")
        unset(__ASAN_FLAGS__)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_link_options("-static-libasan")
        endif()
    else()
        message(FATAL_ERROR "UNSUPPORTED: `PPLNN_USE_SANITIZE` is ON when using compiler `${CMAKE_CXX_COMPILER_ID}`.")
    endif()
endif()

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/x86 DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_x86_static DESTINATION lib)
endif()
