#include <iostream>
#include <ceres/ceres.h>
#include "SnavelyReprojectionError.h"
#include "common.h"//使用其里面定义的BALProblem类读取problem-16-22106-pre.txt内容

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();   //数据归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);   //给数据添加噪声
    bal_problem.WriteToPLYFile("./ch9/inital_ceres.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("./ch9/final_ceres.ply");
    return 0;
}

void SolveBA(BALProblem &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();//return 3; 
    const int camera_block_size = bal_problem.camera_block_size();//return use_quaternions_ ? 10 : 9;
    double * points = bal_problem.mutable_points();//获得待优化系数3d点 points指向3d点的首地址
    double * cameras =  bal_problem.mutable_cameras();//获得待优化系数相机 cameras指向相机的首地址

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double * observations = bal_problem.observations();

    ceres::Problem problem;

    for (size_t i = 0; i < bal_problem.num_observations(); i++)
    {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::create(observations[2*i+0],observations[2*i+1]);//静态函数只要使用类名加范围解析运算符 :: 就可以访问

        // If enabled use Huber's loss function.
        ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);//核函数（损失函数）,参数为阈值

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double * camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double * point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function,loss_function,camera,point);
    }
    
    // show some information here
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;// 设置舒尔消元/边缘化，便于加速求解
    options.minimizer_progress_to_stdout = true;

/****************************************************************************/
//https://blog.csdn.net/hltt3838/article/details/109695164
    //SetOrdering
    ceres::ParameterBlockOrdering * ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras,将偶数索引的空间点变量设为优先消元变量，奇数索引的空间点、相机位姿与内参作为次优先级的消元变量。
    for (size_t i = 0; i < bal_problem.num_points(); i++)
    {
        if(i % 2 == 0)//	取模运算符，整除后的余数
        {
            ordering->AddElementToGroup(points + point_block_size * i,0);
        }
        else
        {
            ordering->AddElementToGroup(points + point_block_size * i,1);
        }
    }

    for (size_t i = 0; i < bal_problem.num_cameras(); i++)
    {
        ordering->AddElementToGroup(cameras + camera_block_size * i,1);
    }
    
    options.linear_solver_ordering.reset(ordering);
/****************************************************************************/
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout << summary.FullReport() << "\n";
}