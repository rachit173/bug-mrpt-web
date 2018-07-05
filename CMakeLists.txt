PROJECT(bug-track)
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
set (CMAKE_CXX_STANDARD 14)  # Require C++14

find_package(MRPT REQUIRED core poses rtti slam topography)

find_package( Threads )

add_subdirectory(rawlog-viewer)
