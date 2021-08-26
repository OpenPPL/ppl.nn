if(PPLNN_INSTALL)
    install(DIRECTORY include DESTINATION .)
    install(TARGETS pplnn_static LIBRARY DESTINATION lib)
endif()
