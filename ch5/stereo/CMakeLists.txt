find_package(Pangolin REQUIRED)

add_executable(stereoVision stereoVision.cpp)
include_directories(${Pangolin_INCLUDE_DIRS})
target_link_libraries(stereoVision ${Pangolin_LIBRARIES})
target_link_libraries(stereoVision ${OpenCV_LIBS})
# target_link_libraries(stereoVision ${OpenCV_LIBS} ${PangolinLIBRARIES})