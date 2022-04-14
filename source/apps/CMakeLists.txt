
#----------------------------------------------------------------------------------------#
#   Add apps
#----------------------------------------------------------------------------------------#


message(STATUS "Adding hicops app...")
add_subdirectory(hicops)

message(STATUS "Adding counter app...")
add_subdirectory(counter)

if (USE_GPU)
    message(STATUS "Adding gicops app...")
    add_subdirectory(gicops)
else()
    message(STATUS "Skipping gicops app... USE_GPU=OFF")
endif()

message(STATUS "Adding test app...")
add_subdirectory(testapp)

message(STATUS "Adding argp app...")
add_subdirectory(argp)
