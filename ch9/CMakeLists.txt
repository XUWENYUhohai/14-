cmake_minimum_required(VERSION 2.8)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

#bundle_adjustment_ceres不需要这个
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
#https://blog.csdn.net/weixin_39393741/article/details/85070299

# SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)
set(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories("/usr/include/eigen3"
                    ${CERES_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR})

# include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(bal_common common.cpp)

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common)

add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS} bal_common)
# target_link_libraries(bundle_adjustment_g2o g2o_csparse_extension g2o_stuff g2o_core cxsparse bal_common)
