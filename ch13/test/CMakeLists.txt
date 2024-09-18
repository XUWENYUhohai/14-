SET(TEST_SOURCES test_triangulation)

#foreach语句
#start 表示起始数，stop 表示终止数，step 表示步长
# foreach(loop_var RANGE start stop [step])
#     ...
# endforeach(loop_var)

FOREACH (test_src ${TEST_SOURCES})
    ADD_EXECUTABLE(${test_src} ${test_src}.cpp)
    TARGET_LINK_LIBRARIES(${test_src} ${THIRD_PARTY_LIBS} myslam)
    ADD_TEST(${test_src} ${test_src})
ENDFOREACH (test_src)
