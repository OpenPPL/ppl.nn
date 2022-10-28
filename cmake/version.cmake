# these variables can be set to the internal value when plugins are embedded into pplnn_static
#   - PPLNN_VERSION_STR: <major>.<minor>.<patch>
#   - PPLNN_COMMIT_STR: a string

if(NOT PPLNN_VERSION_STR)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION_STRING PPLNN_VERSION_STR)
endif()

string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" __PPLNN__UNUSED__ ${PPLNN_VERSION_STR})
unset(__PPLNN__UNUSED__)

set(PPLNN_VERSION_MAJOR ${CMAKE_MATCH_1})
set(PPLNN_VERSION_MINOR ${CMAKE_MATCH_2})
set(PPLNN_VERSION_PATCH ${CMAKE_MATCH_3})

if(NOT PPLNN_COMMIT_STR)
    hpcc_get_git_info(GIT_HASH_OUTPUT PPLNN_COMMIT_STR)
endif()

if(PPLNN_INSTALL)
    set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/utils/version.h.in
        ${__PPLNN_CMAKE_CONFIG_FILE__}
        @ONLY)
    install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION include/ppl/nn/utils)
    unset(__PPLNN_CMAKE_CONFIG_FILE__)
endif()
