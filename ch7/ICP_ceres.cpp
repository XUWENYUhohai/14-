#include <iostream>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,std::vector<cv::KeyPoint> &keypoints_1,std::vector<cv::KeyPoint> &keypoints_2,std::vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p,const cv::Mat &K);

struct ICP_cost
{
    ICP_cost(cv::Point3f pts1 , cv::Point3f pts2) : _pts1(pts1) , _pts2(pts2) {}

    template<typename T>
    bool operator()(const T  * const rotation_vector , const T * const translation_vector , T * residual) const
    {
        T p_transform[3] , p_origin[3] ;
        p_origin[0] = T(_pts2.x);
        p_origin[1] = T(_pts2.y);
        p_origin[2] = T(_pts2.z);

        ceres::AngleAxisRotatePoint(rotation_vector,p_origin,p_transform);

        p_transform[0] += translation_vector[0]; 
        p_transform[1] += translation_vector[1]; 
        p_transform[2] += translation_vector[2]; 

        residual[0] = T(_pts1.x) - p_transform[0];
        residual[1] = T(_pts1.y) - p_transform[1];
        residual[2] = T(_pts1.z) - p_transform[2];

        return true;
    }
    cv::Point3f _pts1;
    cv::Point3f _pts2;
};



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

  double R_ceres[3] = {0} , t_ceres[3] = {3};

  ceres::Problem problem;
  for (size_t i = 0; i < pts1.size(); i++)
  {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ICP_cost,3,3,3>(new ICP_cost(pts1[i],pts2[i])) , nullptr , R_ceres , t_ceres);
  }
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options,&problem,&summary);

  cout << summary.BriefReport() << endl;
    for (auto r : R_ceres)
    {
        cout << r << endl;
    }
    for (auto t : t_ceres)
    {
        cout << t << endl;
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
