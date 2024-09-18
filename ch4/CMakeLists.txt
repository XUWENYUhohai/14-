# cmake_minimum_required(VERSION 2.8)
# project(ch4)


# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus)
include_directories(${Sophus_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(useSophus useSophus.cpp)
#Sophus库只含头文件，所以并不需要
# target_link_libraries(useSophus ${Sophus_LIBRARIES})
# target_link_libraries(useSophus Sophus::Sophus)

add_subdirectory(example)

