
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    "/usr/include/eigen3"
                    ${Sophus_INCLUDE_DIRS})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBRARIES})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBRARIES})