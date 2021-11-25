
project(testapp LANGUAGES CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/testapp.cpp PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# hicops headers are available
add_executable(testapp ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/testapp.cpp)

# add_dependencies(testapp argparse)

# include core/include and generated files
target_include_directories(testapp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../core/include ${CMAKE_BINARY_DIR} ${ARGPARSE_INCLUDE_DIR})

# link appropriate libraries
target_link_libraries(testapp hicops-core ${MPI_LIBRARIES})

set_property(TARGET testapp hicops-core PROPERTY CUDA_ARCHITECTURES 60 70 80)

# installation
install(TARGETS testapp DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(testapp
    PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)
