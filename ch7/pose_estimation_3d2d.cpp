#include <iostream>
//#include <opencv2/opencv.hpp>
// #include <iomanip>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <chrono>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


using namespace std;

void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,vector<cv::KeyPoint> &keypoints_1,vector<cv::KeyPoint> &keypoints_2,vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p,const cv::Mat &K);

typedef vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// BA by gauss-newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose);

// BA by g2O
void bundleAdjustmentG2O(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose);


int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  
  //-- 读取图像
  cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
  cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<cv::KeyPoint> keypoints_1,keypoints_2;
  vector<cv::DMatch> matches;

  find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  cv::Mat d1 = cv::imread(argv[3],CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
  //CV_LOAD_IMAGE_UNCHANGED 读入完整图片
  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<cv::Point3f> pts_3d;
  vector<cv::Point2f> pts_2d;
  for (cv::DMatch m : matches)
  {
    ushort d = d1.ptr<ushort>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if(d == 0) continue;// bad depth

    float dd = d / 5000.0;//?
    
    cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);
    pts_3d.push_back(cv::Point3f(p1.x * dd,p1.y * dd,dd));//这里以第一个相机坐标系为世界坐标系
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }
  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  cv::Mat r,t;
  cv::solvePnP(pts_3d,pts_2d,K,cv::Mat(),r,t,false);// 调用OpenCV 的 PnP 求解(这里默认为LM迭代法：SOLVE_ITERATIVE)，第八个参数可选择EPNP(SOLVEPNP_EPNP)，DLS，UPNP等方法
  //distCoeffs为畸变参数，这里应该为0
  //useExtrinsicGuess若为true则使用提供的r，t为初始近似值进行优化
  cv::Mat R;

  cv::Rodrigues(r,R);// r为旋转向量形式，用Rodrigues公式转换为矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
  cout << "solve pnp in opencv cost time: " << time.count() << " seconds." << endl;
  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;


  // BA by gauss-newton
  VecVector2d pts_2d_eigen;
  VecVector3d pts_3d_eigen;
  for (size_t i = 0; i < pts_3d.size(); i++)
  {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x,pts_3d[i].y,pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x,pts_2d[i].y));
  }
  cout << "calling bundle adjustment by gauss newton" << endl;

  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen,pts_2d_eigen,K,pose_gn);
  t2 = chrono::steady_clock::now();
  time = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "solve pnp by gauss newton cost time: " << time.count() << " seconds." << endl;


  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen,pts_2d_eigen,K,pose_g2o);
  t2 = chrono::steady_clock::now();
  time = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "solve pnp by g2o cost time: " << time.count() << " seconds." << endl;
  return 0;
}



void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,vector<cv::KeyPoint> &keypoints_1,vector<cv::KeyPoint> &keypoints_2,vector<cv::DMatch> &matches)
{
  //-- 初始化
  cv::Mat descriptors_1,descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1,keypoints_1);
  detector->detect(img_2,keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1,keypoints_1,descriptors_1);
  descriptor->compute(img_2,keypoints_2,descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<cv::DMatch> match;
  matcher->match(descriptors_1,descriptors_2,match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000 , max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (size_t i = 0; i < match.size(); i++)
  {
    if(match[i].distance < min_dist) min_dist = match[i].distance;
    if(match[i].distance > max_dist) max_dist = match[i].distance;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (size_t i = 0; i < match.size(); i++)
  {
    if(match[i].distance < max(2*min_dist,30.0))
    { 
      matches.push_back(match[i]);
    }
  }  
}

cv::Point2d pixel2cam(const cv::Point2d &p,const cv::Mat &K)
{
  return(cv::Point2d((p.x - K.at<double>(0,2))/K.at<double>(0,0),(p.y - K.at<double>(1,2))/K.at<double>(1,1)));
}




void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose)
{
  typedef Eigen::Matrix<double,6,1> Vector6d;
  const int iterations = 10;
  double cost = 0,lastCost = 0;
  double fx = K.at<double>(0,0);
  double fy = K.at<double>(1,1);
  double cx = K.at<double>(0,2);
  double cy = K.at<double>(1,2);

  for (size_t iter = 0; iter < iterations; iter++)
  {
    cost = 0;
    Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
    Vector6d b = Vector6d::Zero();

    // compute cost
    for (size_t i = 0; i <points_3d.size(); i++)
    {
      //Sophus::SE3可以直接和Eigen::Vector3相乘
      Eigen::Vector3d pc = pose * points_3d[i];//初始的Sophus::SE3d &pose应该是单位阵
      double inv_z = 1.0 / pc(2);
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(pc(0) * fx / pc(2) + cx , pc(1) * fy / pc(2) + cy);
      Eigen::Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();//矩阵/向量所有元素的平方和

      Eigen::Matrix<double,2,6> J_T;//J_T才是e的导数
      J_T << -fx * inv_z , 0 , fx * pc[0] * inv_z2 , fx * pc[0] * pc[1] * inv_z2 , -fx - fx * pc[0] * pc[0] * inv_z2 , fx * pc[1] * inv_z ,
             0 , -fy * inv_z , fy * pc[1] * inv_z2 , fy + fy * pc[1] * pc[1] * inv_z2 , -fy * pc[0] * pc[1] * inv_z2 , -fy * pc[0] * inv_z; 
      
      H += J_T.transpose() * J_T;
      b += - J_T.transpose() * e;  
    }
    
    Vector6d dx = H.ldlt().solve(b);

    if(isnan(dx(0)))
    {
      cout << "result is nan!" << endl;
      break;
    }

    if(iter > 0 && cost > lastCost)
    {
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    // cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;//std::setprecision(12)需要#include <iomanip>,保留几位小数
    cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;//不需要头文件
    if(dx.norm() < 1e-6)//norm（） == L2范数 == 欧氏距离 == 模
    {
      // converge收敛
      break;
    }
  }
  cout << "pose by g-n: \n" << pose.matrix() << endl;
}



// vertex and edges used in g2o BA
//自定义顶点
// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class VertexPose : public g2o::BaseVertex<6,Sophus::SE3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override
    {
      _estimate = Sophus::SE3d();
    }

    // left multiplication on SE3   左乘
    virtual void oplusImpl(const double * update) override
    {
      Eigen::Matrix<double,6,1> update_eigen;
      update_eigen << update[0] , update[1] , update[2] , update[3] , update[4] , update[5];
      _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override{}
};


//自定义边
// 误差模型 模板参数：观测值维度（_measurement），类型，连接顶点类型
class EdgeProjection : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos , const Eigen::Matrix3d &K) : _pos3d(pos) , _K(K) {}

    virtual void computeError() override
    {
      const VertexPose * v = static_cast<const VertexPose *>(_vertices[0]);//_vertices[0]第一个顶点的指针？
      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d pos_pixel = _K * (T * _pos3d);//Sophus::SE3可以直接和Eigen::Vector3相乘
      pos_pixel /= pos_pixel[2];
      _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override 
    {
      const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d pos_cam = T * _pos3d;
      double fx = _K(0, 0);
      double fy = _K(1, 1);
      double cx = _K(0, 2);
      double cy = _K(1, 2);
      double X = pos_cam[0];
      double Y = pos_cam[1];
      double Z = pos_cam[2];
      double Z2 = Z * Z;

      //其实雅可比可以不写，g2o可以自动求导，但速度会下降
      _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out)const override{}

  private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose)
{
  // 构建图优化，先设定g2o
  //https://www.zhihu.com/question/337617035     https://blog.csdn.net/hhz_999/article/details/123543749
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType;// pose is 6, landmark is 3（points_3d）这里的landmark指的不是误差（边）的维度，而是顶点维度，但因为其是一元边或者说是跟据landmark来定
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> linearSolverType;// 线性求解器类型

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<linearSolverType>()));

  g2o::SparseOptimizer optimizer;// 图模型
  optimizer.setAlgorithm(solver); // 设置求解器
  optimizer.setVerbose(true);// 打开调试输出

  // vertex
  VertexPose * vertex_pose = new VertexPose();// camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());//  优化变量初始化,初始的T为单位阵
  optimizer.addVertex(vertex_pose);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
          K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
          K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int index = 1;
  for (size_t i = 0; i < points_3d.size(); i++)
  {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d,K_eigen);
    edge->setId(index);
    edge->setVertex(0,vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }
  
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  // optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}