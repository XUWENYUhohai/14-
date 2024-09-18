#cmake_minimum_required(VERSION 2.8)
#project(ch6)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBRARIES})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})

# G2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})



add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${G2O_CORE_LIBRARY} g2o_core g2o_stuff)
