
project(argp LANGUAGES C CXX)

# timemory headers are available
add_executable(argp ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/argp.cpp)

add_dependencies(argp argparse)

# include core/include and generated files
target_include_directories(argp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../core/include ${CMAKE_BINARY_DIR} ${MAGICENUM_INCLUDE_DIR} ${ARGPARSE_INCLUDE_DIR})

set_target_properties(argp
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS argp DESTINATION ${CMAKE_INSTALL_BINDIR}/tools)
