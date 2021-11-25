
#----------------------------------------------------------------------------------------#
#   Add apps
#----------------------------------------------------------------------------------------#


message(STATUS "Adding hicops app...")
add_subdirectory(hicops)

message(STATUS "Adding counter app...")
add_subdirectory(counter)

message(STATUS "Adding gicops app...")
add_subdirectory(gicops)

message(STATUS "Adding test app...")
add_subdirectory(testapp)

