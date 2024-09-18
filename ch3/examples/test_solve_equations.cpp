#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>// 包含了Core/Geometry/LU/Cholesky/SVD/QR/Eigenvalues模块

using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 10

//LU分解（上下三角），QR分解（正交阵、上三角），Cholesky分解(LLT,LDLT)（下三角和其自身的共轭转置），SVD（奇异值分解）
int main(int argc, char *argv[])
{
    //生成一个10×10的随机矩阵A
    Matrix<double,MATRIX_SIZE,MATRIX_SIZE> A = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);

    //生成一个10×1的列向量b
    Matrix<double,MATRIX_SIZE,1> b = MatrixXd::Random(MATRIX_SIZE,1);

    //直接求逆
    Matrix<double,MATRIX_SIZE,1> x1 = A.inverse() * b;
    cout<<"直接求逆得到的x为"<<endl;
    cout<<x1.transpose()<<endl;

    //QR分解
    Matrix<double,MATRIX_SIZE,1> x2 = A.householderQr().solve(b);  
    Matrix<double,MATRIX_SIZE,1> x2 = A.colPivHouseholderQr().solve(b);  
    Matrix<double,MATRIX_SIZE,1> x2 = A.fullPivHouseholderQr().solve(b);  
    cout<<"QR分解得到的x为"<<endl;
    cout<<x2.transpose()<<endl; 

    //LU分解
    Matrix<double,MATRIX_SIZE,1> x3 = A.lu().solve(b);
    Matrix<double,MATRIX_SIZE,1> x3 = A.partialPivLu().solve(b);//要求A为可逆的
    Matrix<double,MATRIX_SIZE,1> x3 = A.fullPivLu().solve(b);

    //Cholesky分解
    //LLT分解,此处A不满足正定，故求解出错
    Matrix<double,MATRIX_SIZE,1> x4 = A.llt().solve(b);
    cout<<"LLT分解得到的x为"<<endl;
    cout<<x4.transpose()<<endl;

    //Cholesky分解
    //LDLT分解,此处A不满足半正定或半负定，故求解出错
    Matrix<double,MATRIX_SIZE,1> x5 = A.ldlt().solve(b);
    cout<<"LDLT分解得到的x为"<<endl;
    cout<<x5.transpose()<<endl;

    //SVD分解
    Matrix<double,MATRIX_SIZE,1> x6 = A.bdcSvd().solve(b); 
    Matrix<double,MATRIX_SIZE,1> x6 = A.jacobiSvd().solve(b); 
    return 0;
}
