//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD   与书上178不同： https://blog.csdn.net/weixin_37953603/article/details/108035773
 * @param poses     poses,世界系到相机系的投影矩阵         
 * @param points    points in normalized plane(相机归一化平面坐标)
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {//[R,t] [x,y,z,1] = z [u,v，1] , 因为[x,y,z,1]为非零解所以[R,t]不满秩,且只有唯一解所以r = n-1
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);//<子矩阵大小>（从i行j列开始提取）
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);//若系数矩阵满秩，则齐次线性方程组有且仅有零解，若系数矩阵降秩，则有无穷多解，且基础解系的向量个数等于n-r
//非零奇异值数 = r = n-1 ， 奇异值矩阵阶数 = min（m，n），零奇异值只有一个其对应的奇异值向量必为[x,y,z,1]
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();//除svd.matrixV()(3, 3)是为了变回齐次坐标，因为求出来的可能是基础解系的倍数

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {//svd.singularValues()[3] 应<< svd.singularValues()[2]
        
        return true;
    }
    return false;// 解质量不好，放弃
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
