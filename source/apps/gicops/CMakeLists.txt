
project(gicops LANGUAGES CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/gicops.cpp PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# hicops headers are available
add_executable(gicops ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/gicops.cpp)

add_dependencies(gicops argparse)

# include core/include and generated files
target_include_directories(gicops PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../core/include ${CMAKE_BINARY_DIR} ${ARGPARSE_INCLUDE_DIR})

# link appropriate libraries
target_link_libraries(gicops hicops-core ${MPI_LIBRARIES})

set_property(TARGET gicops hicops-core PROPERTY CUDA_ARCHITECTURES 60 70 80)

# installation
install(TARGETS gicops DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(gicops
    PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)