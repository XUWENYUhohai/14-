#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>//鲁棒核函数    kernel内核   impl实现类（接口）

#include "common.h"
#include <sophus/se3.hpp>

using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics
{
    PoseAndIntrinsics(){}

    /// set from given data address

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    explicit PoseAndIntrinsics(double * data_addr)
    {
        rotation = Sophus::SO3d::exp(Eigen::Vector3d(data_addr[0],data_addr[1],data_addr[2]));
        translation = Eigen::Vector3d(data_addr[3],data_addr[4],data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // 将估计值放入内存
    void set_to(double *data_addr)
    {
        auto r = rotation.log();
        for(int i = 0;i < 3;i++) data_addr[i] = r[i];
        for(int i = 0;i < 3;i++) data_addr[i+3] = translation[i];
        data_addr[6] = focal;  
        data_addr[7] = k1;  
        data_addr[8] = k2;  
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0;
    double k1 = 0 , k2 = 0;
};



/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9,PoseAndIntrinsics>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics(){}

    virtual void setToOriginImpl() override
    {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double * update) override
    {
        _estimate.rotation = Sophus::SO3d::exp(Eigen::Vector3d(update[0],update[1],update[2])) * _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3],update[4],update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 根据估计值投影一个点
    Eigen::Vector2d project(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();//这里有点问题，多了一个1
        double distortion = 1 + r2 * (_estimate.k1 + r2 * _estimate.k2);
        return Eigen::Vector2d(_estimate.focal * distortion * pc[0] , _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in){}
    virtual bool write(ostream &out) const {}
};



class VertexPoint : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint(){}

    virtual void setToOriginImpl() override
    {
        _estimate << 0,0,0;
    }

    virtual void oplusImpl(const double  * update) override
    {
        _estimate += Eigen::Vector3d(update[0],update[1],update[2]);
    }

    virtual bool read(istream &in){}
    virtual bool write(ostream &out) const{}
};



class EdgeProjection : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,VertexPoseAndIntrinsics,VertexPoint>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void computeError() override 
  {
    auto v0 = (VertexPoseAndIntrinsics *)_vertices[0];
    auto v1 = (VertexPoint *)_vertices[1];
    //   const VertexPoseAndIntrinsics * v0 = static_cast<const VertexPoseAndIntrinsics *>(_vertices[0]);
    //   const VertexPoint * v1 = static_cast<const VertexPoint *>(_vertices[1]);
    auto proj = v0->project(v1->estimate());
    _error = proj - _measurement;
  }  

  // use numeric derivatives使用数值导数 

  virtual bool read(istream &in){}
  virtual bool write(ostream &out) const{}
};


void SolveBA(BALProblem &bal_problem);

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "usage: ./" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1,0.5,0.5);
    // bal_problem.WriteToPLYFile("./ch9/inital_g2o_tests.ply");
    bal_problem.WriteToPLYFile("../inital_g2o_tests.ply");//不用vs时，使用终端执行使用
    SolveBA(bal_problem);
    // bal_problem.WriteToPLYFile("./ch9/final_g2o_test.ply");
    bal_problem.WriteToPLYFile("../final_g2o_test.ply");//不用vs时，使用终端执行使用
    return 0;
}


void SolveBA(BALProblem &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double * points = bal_problem.mutable_points();
    double * cameras = bal_problem.mutable_cameras();

    // pose dimension 9, landmark is 3
    // typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3>> BlockSolverType;//这里的landmark=3不是指误差（边）的维度，而是顶点的维度或者说是landmark的维度
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;//https://zhuanlan.zhihu.com/p/338569231
    // g2o在做BA的优化时必须将其所有地图点全部schur掉，否则会出错（ceres无）。
    // 原因是使用了g2o::LinearSolver<BalBlockSolver::PoseMatrixType>这个类型来指定linearsolver,
    // 其中模板参数当中的位姿矩阵类型PoseMatrixType在程序中为相机姿态参数的维度，于是BA当中schur消元后解得线性方程组必须是只含有相机姿态变量。
    //使用g2o::BlockSolverX则不用



    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double * observations = bal_problem.observations();

    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (size_t i = 0; i < bal_problem.num_cameras(); i++)
    {
        VertexPoseAndIntrinsics * v = new VertexPoseAndIntrinsics();
        double * camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));//设置初始值
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    for (size_t i = 0; i < bal_problem.num_points(); i++)
    {
        VertexPoint * v = new VertexPoint();
        double * point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Eigen::Vector3d(point[0],point[1],point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);//前面改为typedef g2o::BlockSolverX BlockSolverType;则不用
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }
    
    // edge
    for (size_t i = 0; i < bal_problem.num_observations(); i++)
    {
        EdgeProjection * edge = new EdgeProjection();
        edge->setVertex(0,vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1,vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Eigen::Vector2d(observations[i*2+0],observations[i*2+1]));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);
    
    // set to bal problem
    //存入内存
    for (size_t i = 0; i < bal_problem.num_cameras(); i++)
    {
        double * camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    
    for (size_t i = 0; i < bal_problem.num_points(); i++)
    {
        double * point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for(int k =0;k < 3;k++) point[k] = vertex->estimate()[k];
        // for(int k =0;k < 3;k++) point[k] = vertex->estimate()(k);
    }
}