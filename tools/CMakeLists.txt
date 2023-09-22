file(GLOB PPLNN_TOOL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/pplnn_llm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)

add_executable(pplnn_llm ${PPLNN_TOOL_SRC})
target_link_libraries(pplnn_llm PRIVATE pplnn_static)
target_compile_features(pplnn_llm PRIVATE cxx_std_11)

if(PPLNN_CUDA_ENABLE_NCCL)
    set(PPLNN_ENABLE_MPI_TOOLS ON)
endif()

if (PPLNN_ENABLE_MPI_TOOLS)
    find_package(MPI)
    if (NOT MPI_CXX_FOUND)
        message(FATAL_ERROR "MPI CXX not found")
    endif()
    target_link_libraries(pplnn_llm PRIVATE ${MPI_CXX_LIBRARIES})
    target_compile_definitions(pplnn_llm PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} PPLNN_ENABLE_MPI_TOOLS)
    target_compile_options(pplnn_llm PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
    target_include_directories(pplnn_llm PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()

find_package(OpenMP REQUIRED)

add_executable(benchmark_llama 
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_llama.cc sampler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)
target_link_libraries(benchmark_llama pplnn_static OpenMP::OpenMP_CXX)
target_include_directories(benchmark_llama PRIVATE ${HPCC_DEPS_DIR}/rapidjson/include)