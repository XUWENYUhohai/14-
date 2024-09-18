//
// Created by gaoxiang on 19-5-4.
//
#include <gtest/gtest.h>
#include "myslam/common_include.h"
#include "myslam/algorithm.h"
//http://www.360doc.com/content/20/1118/20/65839755_946572358.shtml   ,    http://www.uml.org.cn/Test/201905061.asp
TEST(MyslamTest, Triangulation) {
    Vec3 pt_world(30, 20, 10), pt_world_estimated;
    std::vector<SE3> poses{
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, 0, 0)),
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, -10, 0)),
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, 10, 0)),
    };
    std::vector<Vec3> points;
    for (size_t i = 0; i < poses.size(); ++i) {
        Vec3 pc = poses[i] * pt_world;
        pc /= pc[2];
        points.push_back(pc);
    }

    EXPECT_TRUE(myslam::triangulation(poses, points, pt_world_estimated));//EXPECT宏当检测失败后继续向下执行，assert宏检测失败后退出当前函数
    EXPECT_NEAR(pt_world[0], pt_world_estimated[0], 0.01);//https://zhuanlan.zhihu.com/p/324639477
    EXPECT_NEAR(pt_world[1], pt_world_estimated[1], 0.01);///期待返回值pt_world[1]，在pt_world_estimated[1] +- 0.01之间
    EXPECT_NEAR(pt_world[2], pt_world_estimated[2], 0.01);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);//前面加::与局部变量区分开，是全局变量
    return RUN_ALL_TESTS();
}