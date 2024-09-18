#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>//vertex顶点
#include <g2o/core/base_unary_edge.h>//unary一元
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
//#include <memory>

using namespace std;

//https://blog.csdn.net/qq_42722197/article/details/122356361?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-122356361-null-null.pc_agg_new_rank&utm_term=setToOriginImpl%28%29&spm=1000.2123.3001.4430
//自定义顶点
// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW//EIGEN_MAKE_ALIGNED_OPERATOR_NEW声明在new一个这样类型的对象时，实现内存对齐，https://www.dazhuanlan.com/ultralisk/topics/1413293

    //重置顶点，设定顶点初始值
    //https://wenku.baidu.com/view/134f0cd10142a8956bec0975f46527d3240ca6da.html,override明确告诉编译器需要重写虚函数
    virtual void setToOriginImpl() override
    {
        _estimate << 0,0,0;
    }

    //更新
    virtual void oplusImpl(const double * update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘：留空(声明一下就可以)
    virtual bool read(istream &in){}
    virtual bool write(ostream & out) const {}
};


//自定义边
// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(),_x(x) {}//子类有参构造时必须要先调用父类构造，因为会沿用父类的东西(没被覆盖的函数以及可见的成员变量等)

    // 计算曲线模型误差
    virtual void computeError() override
    {//观测值存储在_measurement 中，误差存储在_error 中，节点存储在_vertices[] 中

        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);//获取节点（顶点）
        const Eigen::Vector3d abc = v->estimate();// 获取节点的优化变量（节点值）
        _error(0,0) = _measurement - std::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    }

    // 计算雅可比矩阵
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex * v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = - exp(abc(0) * _x * _x + abc(1) * _x + abc(2));
        //double y = - exp(abc(0,0) * _x * _x + abc(1,0) * _x + abc(2,0));

        _jacobianOplusXi[0] = _x * _x * y;
        _jacobianOplusXi[1] = _x * y;
        _jacobianOplusXi[2] = y; 
    }

    virtual bool read(istream & in){}
    virtual bool write(ostream & out) const{}


    double _x;  // x 值， y 值为 _measurement
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


    // 构建图优化，先设定g2o
    // typedef g2o::BlockSolverX BlockSolverType;
    //https://www.zhihu.com/question/337617035     https://blog.csdn.net/hhz_999/article/details/123543749
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;// 每个误差项优化变量维度为3，（误差值维度为1）这里的1不一定指的是误差维度，只是这里为一元边，正常g2o::BlockSolverTraits<PoseDim,LandMarkDim>是所定义边的两个顶点的维度，也可以说第二个是跟据landmark来定

    //方法1
    //线性求解器的类型可选LinearSolverDense、LinearSolverPCG、LinearSolverCSparse、LinearSolverCholmod
    //typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;// 线性求解器类型

    // 创建总求解器solver：梯度下降方法，可以从GN, LM, DogLeg 中选
    // auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    //     g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    //g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));


        //std::unique_str为一种智能指针（<memory>）它持有对对象的所有权，即两个unique_str不能指向一个对象，其指向的堆内存无法与其他unique_str共享，不能进行拷贝操作，只能移动
        //当他指向其他对象时，之前所指对象会被销毁；超出作用域时也会被销毁
        //std::unique_str<int> p1(new int(5));
        //std::unique_str<int> p2 = p1或p2(p1)//编译出错
        //std::unique_str<int> p3 = std::move(p1)或p3(std::move(p1))转移所有权，内存归p3,p1为无效指针
        //p3.reset();释放内存
        //p3.release();释放p3所有权，但存储空间不会被销毁
        //p1.reset();无效
        //p3.rest(p);p为普通指针，p3获取p的内存和所有权


    //方法2
    BlockSolverType::LinearSolverType  * LinearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();/*创建线性求解器*/
    BlockSolverType * solver_ptr = new BlockSolverType(std::unique_ptr<BlockSolverType::LinearSolverType>(LinearSolver));/*创建BlockSolver*/
    g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<BlockSolverType>(solver_ptr));/*创建总求解器solver*/


    //Sparse稀疏
    g2o::SparseOptimizer optimizer;//  图模型
    optimizer.setAlgorithm(solver);// 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex * v = new CurveFittingVertex();//  实例化节点类型
    v->setEstimate(Eigen::Vector3d(ae,be,ce));//  优化变量初始化
    v->setId(0);//  设置图中节点id
    optimizer.addVertex(v);//  将节点加入图模型

    // 往图中增加边
    for (size_t i = 0; i < N; i++)
    {
        CurveFittingEdge * edge = new CurveFittingEdge(x_data[i]);  //  实例化边类型，传入特征x
        edge->setId(i);
        //edge->setVertex(0,v);    //  设置连接的节点：这个边连接的第几个节点编号和节点对象
        edge->setVertex(0,optimizer.vertices()[0]);

        edge->setMeasurement(y_data[i]); // 观测数值
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity() * 1 / w_sigma * w_sigma);// 信息矩阵：协方差矩阵之逆
        
        optimizer.addEdge(edge);
    }
    
    // 执行优化
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();//  初始化优化
    optimizer.optimize(10);//  优化次数设置
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;
    
    return 0;
}
