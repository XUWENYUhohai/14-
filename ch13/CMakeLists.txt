cmake_minimum_required(VERSION 2.8)
project(myslam)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)#设置依赖包的文件夹，添加cmake_modules到CMAKE_MODULE_PATH路径列表中
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)#设置可执行文件生成位置到bin
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)#设置库文件生成路径到lib

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED) #寻找库
include_directories(${OpenCV_INCLUDE_DIRS})#添加头文件 

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)#http://www.360doc.com/content/20/1118/20/65839755_946572358.shtml   ,    http://www.uml.org.cn/Test/201905061.asp
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)#https://zhuanlan.zhihu.com/p/95889124  ,    https://blog.csdn.net/u013066730/article/details/84103083
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        #https://blog.csdn.net/shixin_0125/article/details/103966355   ,  https://blog.csdn.net/shixin_0125/article/details/103966355
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        # 所有G2O?  :  SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
        ${GTEST_BOTH_LIBRARIES}
        # ${GTEST_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        #https://www.zhihu.com/question/36236334  ,  https://blog.csdn.net/slslslyxz/article/details/110392528
        pthread
        ${CSPARSE_LIBRARY}
        )

enable_testing()# 使用 enable_testing() 命令可以启用测试。一定要在根目录下的CMakeLists.txt中开启，不然执行make test时会报错。  https://blog.csdn.net/weixin_39766005/article/details/122686185

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)      #添加自己写的头文件 
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
