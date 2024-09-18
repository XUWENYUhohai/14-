cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)
find_package(Ceres REQUIRED)
find_package(CSparse REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${Sophus_INCLUDE_DIRS}
                    "/usr/include/eigen3"
                    ${G2O_INCLUDE_DIRS}
                    ${Ceres_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR})

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBRARIES})

add_executable(test_feature_points test_feature_points.cpp)
target_link_libraries(test_feature_points ${OpenCV_LIBRARIES})

add_executable(test_match test_match.cpp)
target_link_libraries(test_match ${OpenCV_LIBRARIES})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBRARIES})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBRARIES})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBRARIES} g2o_core g2o_stuff)

add_executable(pose_estimation_3d3d_add_3dpoints pose_estimation_3d3d_add_3dpoints.cpp)
target_link_libraries(pose_estimation_3d3d_add_3dpoints ${OpenCV_LIBRARIES} g2o_core g2o_stuff g2o_csparse_extension cxsparse)
# 所有G2O?  :  SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

add_executable(PnP_ceres PnP_ceres.cpp)
target_link_libraries(PnP_ceres ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

add_executable(ICP_ceres ICP_ceres.cpp)
target_link_libraries(ICP_ceres ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})