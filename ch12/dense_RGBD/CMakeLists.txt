cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pcl 
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})#https://www.cnblogs.com/sunbines/p/16155640.html

# octomap 
find_package(Octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(surfel_mapping surfel_mapping.cpp)
target_link_libraries(surfel_mapping ${PCL_LIBRARIES})


add_executable(octomap_mapping octomap_mapping.cpp)
target_link_libraries(octomap_mapping ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES})