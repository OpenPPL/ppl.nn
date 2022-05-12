# these variables can be set to the internal value when plugins are embedded into pplnn_static
#   - PPLNN_VERSION_STR: <major>.<minor>.<patch>
#   - PPLNN_COMMIT_HASH: a string

if(NOT PPLNN_VERSION_STR)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION PPLNN_VERSION_STR)
endif()

string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" __PPLNN__UNUSED__ ${PPLNN_VERSION_STR})
set(PPLNN_VERSION_MAJOR ${CMAKE_MATCH_1})
set(PPLNN_VERSION_MINOR ${CMAKE_MATCH_2})
set(PPLNN_VERSION_PATCH ${CMAKE_MATCH_3})

list(APPEND PPLNN_COMPILE_DEFINITIONS
    PPLNN_VERSION_MAJOR=${PPLNN_VERSION_MAJOR}
    PPLNN_VERSION_MINOR=${PPLNN_VERSION_MINOR}
    PPLNN_VERSION_PATCH=${PPLNN_VERSION_PATCH})

if(NOT PPLNN_COMMIT_STR)
    hpcc_get_git_info(GIT_HASH_OUTPUT PPLNN_COMMIT_STR)
endif()

set(__PPLNN_COMMIT_ID_SRC__ ${CMAKE_CURRENT_BINARY_DIR}/generated/commit.cc)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/utils/commit.cc.in
    ${__PPLNN_COMMIT_ID_SRC__}
    @ONLY)

list(APPEND PPLNN_SOURCES ${__PPLNN_COMMIT_ID_SRC__})

unset(__PPLNN_COMMIT_ID_SRC__)
unset(__PPLNN__UNUSED__)
