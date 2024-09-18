#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 代价函数的计算模型
struct CURVE_FITTING_COST//https://www.guyuehome.com/37349
{
    CURVE_FITTING_COST(double x,double y) : _x(x),_y(y){}

    // 残差的计算
    //得到一个可以自动求导的代价函数,必须定义一个类或者结构体在里面重载运算符,在里面实现用参数模板计算代价函数,重载的运算符必须在最后一个参数里存入计算结果,并且返回true.
    template<typename T>
    bool operator()(
        //比如int const*a;，实际上可以看成是int const (*a)，这表示指针a所指向的地址可以变，但是所指向的那个值不能变。而int *const a;，可以看成int* (const a);
        //我们都知道a的值其实是一个地址，这就表示a所保存的地址是不可以变的，但是这个地址对应的值是可以变的。
        const T *const abc,// 模型参数，有3维,需要优化的量
        T *residual) const
    {
        // y-exp(ax^2+bx+c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x,_y;
};



int main(int argc, char *argv[])
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;      // 数据

    for (size_t i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma));
    }
    
    double abc[3] = {ae,be,ce};//优化变量

    // 构建最小二乘问题
    ceres::Problem problem;//代表具有双边约束的最小二乘问题
    for (size_t i = 0; i < N; i++)
    {//https://blog.csdn.net/u014709760/article/details/88091720/

        problem.AddResidualBlock(// 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(new CURVE_FITTING_COST(x_data[i],y_data[i])), //这个输出上面那个结果体或类的指针，这其中完成了残差和J的计算
        nullptr,// 核函数(损失函数)，这里不使用，为空
        abc// 待估计参数，优化变量
        );
    }
    
    // 配置求解器
    ceres::Solver::Options options;// 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;// 增量方程如何求解
    options.minimizer_progress_to_stdout = true;// 输出到cout
 
    ceres::Solver::Summary summary; // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options,&problem,&summary); //开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";     
    for (auto a : abc)
    {
        cout << a << " ";
    }
    cout << endl;  

    return 0;                                                            
}


