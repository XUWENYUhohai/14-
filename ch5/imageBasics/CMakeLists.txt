project(imageBasics)

add_executable(imageBasics imageBasics.cpp)

# 链接OpenCV库
# target_link_libraries(imageBasics ${OpenCV_LIBRARIES})
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(undistortImage ${OpenCV_LIBRARIES})