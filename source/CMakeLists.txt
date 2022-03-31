
#----------------------------------------------------------------------------------------#
#   hicops source
#----------------------------------------------------------------------------------------#

message(STATUS "Adding core...")
add_subdirectory(core)

message(STATUS "Adding hicops app..")
add_subdirectory(app)

message(STATUS "Adding argp app...")
add_subdirectory(argp)

#----------------------------------------------------------------------------------------#
#   counter source
#----------------------------------------------------------------------------------------#
message(STATUS "Adding counter app...")
add_subdirectory(counter)
