#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

/// 本程序演示sophus的基本用法
int main(int argc, char *argv[])
{
    // 沿Z轴转90度的旋转矩阵
    Matrix3d R = AngleAxisd(M_PI/2,Vector3d(0,0,1)).matrix();
    //Matrix3d R = AngleAxisd(M_PI/2,Vector3d(0,0,1)).toRotationMatrix();
    // 或者四元数
    Quaterniond q(R);

    //李群so3
    Sophus::SO3d so3_R(R);// Sophus::SO3d可以直接从旋转矩阵构造
    Sophus::SO3d so3_q(q);// 也可以通过四元数构造
    // 二者是等价的
    cout << "SO(3) from matrix:\n" << so3_R.matrix() << endl;
    cout << "SO(3) from quaternion:\n" << so3_q.matrix() << endl;
    cout << "they are equal" << endl;


    // 使用对数映射获得它的李代数
    Vector3d so3 = so3_R.log();
    cout << "so3 = " << so3.transpose() << endl;

    //hat为向量(李代数)到反对称矩阵
    cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) <<endl;
    // 相对的，vee为反对称到向量
    cout << "so3 vee=\n" << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

    double theta1 = acos((R.trace()-1)/2);
    double theta2 = so3.norm();//求取so3的模值就是theta
    cout << theta1 << "==" << theta2 << endl;

    //J的求法，BCH近似公式
    Matrix3d J = sin(theta1) / theta1 * Eigen::Matrix3d::Identity() + (1 - sin(theta1)/theta1) * so3 / theta1 * (so3 / theta1).transpose() + (1 - cos(theta1)) / theta1 * Sophus::SO3d::hat(so3/theta1); 
    cout<<"J="<<J<<endl;

    // 增量扰动模型的更新
    Vector3d update_so3(1e-4,0,0);//假设更新量为这么多
    Sophus::SO3d so3_updated = Sophus::SO3d::exp(update_so3) * so3_R;//得到更新后的李群
    cout << "SO3 updated = \n" << so3_updated.matrix() << endl;
    // Sophus::SO3d so3_updated = Sophus::SO3d::exp(update_so3) * Sophus::SO3d::exp(so3);
    // cout << "SO3 updated = \n" << so3_updated.matrix() << endl;

    cout << "*******************************" << endl;
    // 对SE(3)操作大同小异
    Vector3d t(1,0,0);// 沿X轴平移1
    Sophus::SE3d se3_Rt(R,t);// 从R,t构造SE(3)
    Sophus::SE3d se3_qt(q,t); // 从q,t构造SE(3)
    cout << "SE3 from R,t= \n" << se3_Rt.matrix() << endl;//李群SE3
    cout << "SE3 from q,t= \n" << se3_qt.matrix() << endl;

    // 李代数se(3) 是一个六维向量，方便起见先typedef一下
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = se3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;

    cout << "rotationMatrix():" << se3_Rt.rotationMatrix() << endl;//R
    cout << "so3():" << se3_Rt.so3().matrix() << endl;//R
    cout << "translation():" << se3_Rt.translation() << endl;//t


    // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.
    // 同样的，有hat和vee两个算符
    cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 vee = \n" << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // 最后，演示一下更新
    Vector6d update_se3;//更新量
    update_se3.setZero();
    update_se3(0,0) = 1e-4;
    Sophus::SE3d se3_updated = Sophus::SE3d::exp(update_se3) * se3_Rt;//得到更新后的李群
    cout << "SE3 updated = " << endl << se3_updated.matrix() << endl;

    Sophus::SE3d test = Sophus::SE3d();
    cout << "test" << test.matrix();

    return 0;
}


