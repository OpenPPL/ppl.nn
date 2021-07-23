find_package(OpenCV QUIET)
if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Samples need opencv will not be built!")
    return()
endif()

add_executable(classification classification.cpp)
target_include_directories(classification PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(classification PUBLIC pplnn_static ${OpenCV_LIBS})
