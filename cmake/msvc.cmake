if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    # for cmake version < 3.15
    foreach(lang C CXX)
        string(REPLACE /MD /MT CMAKE_${lang}_FLAGS_DEBUG "${CMAKE_${lang}_FLAGS_DEBUG}")
        string(REPLACE /MD /MT CMAKE_${lang}_FLAGS_RELEASE "${CMAKE_${lang}_FLAGS_RELEASE}")
    endforeach()
    # for cmake version >= 3.15
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # for cmake version >= 3.15
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
