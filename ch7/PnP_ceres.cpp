#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>//提供了轴角旋转（也就是通过李代数旋转）的函数ceres::AngleAxisRotatePoint()

using namespace std;

void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,vector<cv::KeyPoint> &keypoints_1,vector<cv::KeyPoint> &keypoints_2,vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p,const cv::Mat &K);

struct PnPCost
{
    PnPCost(cv::Point2f pts_2d , cv::Point3f pts_3d) : _pts_2d(pts_2d),_pts_3d(pts_3d) {}

    ////首先根据相机的外参将世界坐标系转换到相机坐标系下并归一化
    template<typename T>
    bool operator()(const T * const rotation_vector , const T * const translation_vector , T * residual) const
    {
        T p_transform[3] , p_origin[3];
        
        p_origin[0] = T(_pts_3d.x);
        p_origin[1] = T(_pts_3d.y);
        p_origin[2] = T(_pts_3d.z);
        // 将世界坐标系的3D点point，转到相机坐标系下的3D点P_transform
        ceres::AngleAxisRotatePoint(rotation_vector,p_origin,p_transform);//https://blog.csdn.net/u013019296/article/details/121506748

        //旋转后加上平移向量
        p_transform[0] += translation_vector[0]; 
        p_transform[1] += translation_vector[1]; 
        p_transform[2] += translation_vector[2]; 

        //归一化
        T xp = p_transform[0] / p_transform[2];
        T yp = p_transform[1] / p_transform[2];

        double fx=520.9, fy=521.0, cx=325.1, cy=249.7;
        T predicteed_x = fx * xp + cx;
        T predicteed_y = fy * yp + cy;

        residual[0] = T(_pts_2d.x) - predicteed_x;
        residual[1] = T(_pts_2d.y) - predicteed_y;

        return true;
    }


    cv::Point2f _pts_2d;
    cv::Point3f _pts_3d;
};



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

    double R_ceres[3] = {0};
    double t_ceres[3] = {0};
    
    ceres::Problem problem;

    for (size_t i = 0; i < pts_3d.size(); i++)
    {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PnPCost,2,3,3>(new PnPCost(pts_2d[i],pts_3d[i])) , nullptr , R_ceres , t_ceres);
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