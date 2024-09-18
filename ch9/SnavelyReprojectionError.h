#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H
/*
该文件给出了投影误差模型
*/
#include <iostream>
#include "ceres/ceres.h"

#include "rotation.h"

class SnavelyReprojectionError
{
private:
    double observed_x;
    double observed_y;
public:
    SnavelyReprojectionError(double observation_x,double observation_y) : observed_x(observation_x),observed_y(observation_y) {}


    template<typename T>
    bool operator()(const T * const camera , const T * const point , T * residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        // 带畸变系数的相机投影过程得到路标点在图像上的像素坐标估计值
        CamProjectionWithDistortion(camera,point,predictions);
         // 估计值与观测值的误差
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T * camera , const T * point , T * predictions)
    {
        // Rodrigues' formula罗德里格斯公式
        T p[3];
        AngleAxisRotatePoint(camera,point,p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];//bal数据集投影时假设投影平面在相机光心之后，要乘以-1
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion // 应用二阶和四阶径向畸变系数
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + r2 * l2);

        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }
 // 静态函数Create作为外部调用接口，直接返回一个可自动求导的Ceres代价函数
    static ceres::CostFunction *create(const double observed_x , const double observed_y)//静态函数只要使用类名加范围解析运算符 :: 就可以访问
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(new SnavelyReprojectionError(observed_x,observed_y)));
    }
};


#endif