include_directories("usr/include/eigen3")

#添加Pangolin依赖
#find_package(Pangolin REQUIRED)
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})#Pangolin库含头文件和源文件，所以需要