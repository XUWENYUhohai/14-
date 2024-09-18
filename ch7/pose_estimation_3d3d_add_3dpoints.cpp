#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>//LM
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <chrono>
#include <sophus/se3.hpp>

using namespace std;

void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,std::vector<cv::KeyPoint> &keypoints_1,std::vector<cv::KeyPoint> &keypoints_2,std::vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p,const cv::Mat &K);

void bundleAdjustment(const vector<cv::Point3f> &pts1 ,vector<cv::Point3f> &pts2 , cv::Mat &R , cv::Mat &t);

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  
  //-- 读取图像
  cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);// 深度图为16位无符号数，单通道图像
  cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);// 深度图为16位无符号数，单通道图像

  vector<cv::KeyPoint> keypoints_1, keypoints_2;
  vector<cv::DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 建立3D点
  cv::Mat depth1 = cv::imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat depth2 = cv::imread(argv[4],CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<cv::Point3f> pts1,pts2;

  for (cv::DMatch m : matches)
  {
    ushort d1 = depth1.ptr<ushort>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<ushort>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if(d1 == 0 || d2 == 0) continue;

    cv::Point2f P1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);
    cv::Point2f P2 = pixel2cam(keypoints_2[m.trainIdx].pt,K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(cv::Point3f(P1.x * dd1 , P1.y * dd1 , dd1));
    pts2.push_back(cv::Point3f(P2.x * dd2 , P2.y * dd2 , dd2));
  }
  
  cout << "3d-3d pairs: " << pts1.size() << endl;
  cv::Mat R,t;


  cout << "calling bundle adjustment" << endl;
  bundleAdjustment(pts1,pts2,R,t);

  // verify p1 = R * p2 + t
  for(int i = 0;i < 5;i++)
  {
    cout << "p1 = " << pts1[i] << endl;
    cout << "p2 = " << pts2[i] << endl;
    cout << "(R*p2+t) = " << R * (cv::Mat_<double>(3,1) << pts2[i].x , pts2[i].y , pts2[i].z) + t << endl;
  }
  return 0;
}



void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,std::vector<cv::KeyPoint> &keypoints_1,std::vector<cv::KeyPoint> &keypoints_2,std::vector<cv::DMatch> &matches)
{
  //-- 初始化
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  detector->detect(img_1,keypoints_1);
  detector->detect(img_2,keypoints_2);

  descriptor->compute(img_1,keypoints_1,descriptors_1);
  descriptor->compute(img_2,keypoints_2,descriptors_2);

  vector<cv::DMatch> match;
  matcher->match(descriptors_1,descriptors_2,match);

  double min_dist = 10000 , max_dist = 0;
  for (size_t i = 0; i < match.size() ; i++)
  {
    if(match[i].distance < min_dist) min_dist = match[i].distance;
    if(match[i].distance > max_dist) max_dist = match[i].distance;
  }
  
  printf("-- Max dist : %f \n",max_dist);
  printf("-- Min dist : %f \n",min_dist);

  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if(match[i].distance <= max(2 * min_dist , 30.0)) matches.push_back(match[i]);
  }
}

cv::Point2d pixel2cam(const cv::Point2d &p , const cv::Mat &K)
{
  return(cv::Point2f((p.x - K.at<double>(0,2)) / K.at<double>(0,0) , (p.y - K.at<double>(1,2)) / K.at<double>(1,1)));
}




// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6,Sophus::SE3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl()override
  {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double * update) override
  {
    Eigen::Matrix<double,6,1> update_eigen;
    update_eigen << update[0] , update[1], update[2], update[3], update[4], update[5];
    _estimate  = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out)const override {}
};

/// 新增的空间点顶点
class VertexPoint : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override
    {
        _estimate = Eigen::Vector3d::Zero();
    }

    virtual void oplusImpl(const double * update) override
    {
        _estimate += Eigen::Vector3d(update[0],update[1],update[2]);
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out)const override {}
};



/// g2o binary edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexPose,VertexPoint>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly() {}

  virtual void computeError() override 
  {
    const VertexPose * pose = static_cast<const VertexPose *>(_vertices[0]);
    const VertexPoint * point = static_cast<const VertexPoint *>(_vertices[1]);
    _error = _measurement - pose->estimate() * point->estimate();
  }

  virtual void linearizeOplus() override 
  {
    VertexPose * pose = static_cast<VertexPose *>(_vertices[0]);
    VertexPoint * point = static_cast<VertexPoint *>(_vertices[1]);
    Eigen::Vector3d xyz_trans = pose->estimate() * point->estimate();
    _jacobianOplusXi.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3,3>(0,3) = Sophus::SO3d::hat(xyz_trans);//Sophus::SE3d::hat() == ^ == 向量->矩阵

    _jacobianOplusXj = -pose->estimate().rotationMatrix();//其实雅可比可以不写，g2o可以自动求导，但速度会下降
  }

  bool read(istream &in){}
  bool write(ostream &out) const {}

};

void bundleAdjustment(const vector<cv::Point3f> &pts1, vector<cv::Point3f> &pts2,cv::Mat &R,cv::Mat &t)
{
  // 构建图优化，先设定g2o
  //https://www.zhihu.com/question/337617035     https://blog.csdn.net/hhz_999/article/details/123543749
  // typedef g2o::BlockSolverX Block;
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;//pose is 6, landmark is 3（points_3d）这里的landmark指的不是误差（边）的维度，而是顶点维度
  typedef g2o::LinearSolverDense<Block::PoseMatrixType> LinearSolverType;
  // typedef g2o::LinearSolverCSparse<Block::PoseMatrixType> LinearSolverType;

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<Block>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;// 图模型
  optimizer.setAlgorithm(solver);// 设置求解器
  optimizer.setVerbose(true);// 打开调试输出

  // vertex1
  VertexPose * pose = new VertexPose();
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  // vertex2
  vector<VertexPoint *> vertex_points;
  for (size_t i = 0; i < pts2.size(); i++)
  {
    VertexPoint * point = new VertexPoint();
    point->setId(i+1);
    point->setEstimate(Eigen::Vector3d(pts2[i].x,pts2[i].y,pts2[i].z));
    point->setMarginalized(true);//G2O中对路标点设置边缘化(pPoint->setMarginalized(true))是为了在计算求解过程中，先消去路标点变量，实现先求解相机位姿，然后再利用求解出来的相机位姿反过来计算路标点的过程，目的是为了加速求解，并非真的将路标点给边缘化掉
    //在使用g2o::BlockSolver<g2o::BlockSolverTraits<6,3>>时需要，使用g2o::BlockSolverX好像不需要
    optimizer.addVertex(point);
    vertex_points.push_back(point);
  }
  
  


  // edges
  for (size_t i = 0; i < pts1.size(); i++)
  {
    EdgeProjectXYZRGBDPoseOnly * edge = new EdgeProjectXYZRGBDPoseOnly();
    edge->setId(i);
    edge->setVertex(0,pose);
    edge->setVertex(1,vertex_points[i]);
    edge->setMeasurement(Eigen::Vector3d(pts1[i].x,pts1[i].y,pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }
  
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (cv::Mat_<double>(3,3) << R_(0, 0), R_(0, 1), R_(0, 2),
                                R_(1, 0), R_(1, 1), R_(1, 2),
                                R_(2, 0), R_(2, 1), R_(2, 2));
  t = (cv::Mat_<double>(3,1) << t_(0),t_(1),t_(2));

  for (size_t i = 0; i < 5; i++)
  {
    pts2[i] = cv::Point3f(vertex_points[i]->estimate()[0],vertex_points[i]->estimate()[1],vertex_points[i]->estimate()[2]);
    // pts2[i] = cv::Point3f(vertex_points[i]->estimate()(0),vertex_points[i]->estimate()(1),vertex_points[i]->estimate()(2));
    cout << "after optimization p2=" << pts2[i] << endl; 
  }
  
}