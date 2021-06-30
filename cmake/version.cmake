hpcc_get_git_info(
    GIT_HASH_OUTPUT PPLNN_COMMIT_HASH
    GIT_TAG_OUTPUT PPLNN_COMMIT_TAG)

set(PPLNN_VERSION_SRC ${CMAKE_CURRENT_BINARY_DIR}/generated/version.cc)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/utils/version.cc.in
    ${PPLNN_VERSION_SRC}
    @ONLY)

list(APPEND PPLNN_SOURCES ${PPLNN_VERSION_SRC})
