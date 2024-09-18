#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,
                          vector<cv::KeyPoint> &keypoints_1,vector<cv::KeyPoint> &keypoints_2,
                          vector<cv::DMatch> &matches);

void pose_estimation_2d2d(const vector<cv::KeyPoint> &keypoints_1,const vector<cv::KeyPoint> &keypoints_2,
                          const vector<cv::DMatch> &matches,
                          cv::Mat &R,cv::Mat &t);

void triangulation(const vector<cv::KeyPoint> &keypoints_1,const vector<cv::KeyPoint> &keypoints_2,
                  const vector<cv::DMatch> &matches,
                  const cv::Mat &R,const cv::Mat &t,
                  vector<cv::Point3d> &points);

// 作图用,储存rgb像素颜色
inline cv::Scalar get_color(float depth)//内联函数:可以解决一些频繁调用的函数大量消耗栈空间（栈内存）的问题。关键字inline必须与函数定义放在一起才能使函数成为内联函数，仅仅将inline放在函数声明前面不起任何作用
{
  float up_th = 50 , low_th = 10 , th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range , 0 , 255 * (1 - depth / th_range));
}

// 像素坐标转相机归一化坐标
cv::Point2f pixel2cam(const cv::Point2d &p,const cv::Mat &K);

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "usage: triangulation img1 img2" << endl;
    return 1;
  }
  
  //-- 读取图像
  cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
  cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);

  vector<cv::KeyPoint> keypoints_1,keypoints_2;
  vector<cv::DMatch> matches;
  find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  //-- 估计两张图像间运动
  cv::Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //-- 三角化
  vector<cv::Point3d> points;
  triangulation(keypoints_1,keypoints_2,matches,R,t,points);//points是相机1下的坐标

  //-- 验证三角化点与特征点的重投影关系
  cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  cv::Mat img1_plot = img_1.clone();
  cv::Mat img2_plot = img_2.clone();

  for (size_t i = 0; i < matches.size(); i++)
  {
    // 第一个图
    float depth1 = points[i].z;
    cout << "depth: " << depth1 << endl;
    cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt,K);

    cv::circle(img1_plot,keypoints_1[matches[i].queryIdx].pt,2,get_color(depth1),2);//https://blog.csdn.net/jinwanchiji/article/details/78925290

    // 第二个图
    cv::Mat pt2_trans = R * (cv::Mat_<double>(3,1) << points[i].x,points[i].y,points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2,0);
    cv::circle(img2_plot,keypoints_2[matches[i].trainIdx].pt,2,get_color(depth2),2);
  }

  cv::imshow("img 1",img1_plot);
  cv::imshow("img 2",img2_plot);
  cv::waitKey();
  return 0;
}




void find_feature_matches(const cv::Mat &img_1,const cv::Mat &img_2,vector<cv::KeyPoint> &keypoints_1,vector<cv::KeyPoint> &keypoints_2,vector<cv::DMatch> &matches)
{
  //-- 初始化
  cv::Mat descriptors_1, descriptors_2;
  // used in OpenCV3
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
  double min_dist = 10000, max_dist = 0;
  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (size_t i = 0; i < match.size(); i++)
  {
    if (match[i].distance > max_dist) max_dist = match[i].distance;
    if (match[i].distance < min_dist) min_dist = match[i].distance;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (size_t i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= max(2 * min_dist , 30.0))
    {
      matches.push_back(match[i]);
    }
  }
}





void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &keypoints_1,const std::vector<cv::KeyPoint> &keypoints_2,const std::vector<cv::DMatch> &matches,cv::Mat &R, cv::Mat &t)
{
  // 相机内参,TUM Freiburg2
  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<cv::Point2f> points1;
  vector<cv::Point2f> points2;

  for (size_t i = 0; i < matches.size(); i++)
  {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }
  
  //-- 计算本质矩阵
  cv::Point2d principal_point(325.1, 249.7);        //相机主点, TUM dataset标定值
  int focal_length = 521;            //相机焦距, TUM dataset标定值
  cv::Mat essential_matrix;
  essential_matrix = cv::findEssentialMat(points1,points2,focal_length,principal_point);

  //-- 从本质矩阵中恢复旋转和平移信息.
  cv::recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);
}




void triangulation(const vector<cv::KeyPoint> &keypoint_1,const vector<cv::KeyPoint> &keypoint_2,
                    const vector<cv::DMatch> &matches,
                    const cv::Mat &R,const cv::Mat &t,
                    vector<cv::Point3d> &points)
{
  cv::Mat T1 = (cv::Mat_<float>(3,4) << 1, 0, 0, 0,////这里以第一张图片为世界坐标系
                                        0, 1, 0, 0,
                                        0, 0, 1, 0);

  cv::Mat T2 = (cv::Mat_<float>(3,4)<< R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                                       R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                                       R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  vector<cv::Point2f> pts_1,pts_2;
  for ( cv::DMatch m : matches)
  {
    // 将像素坐标转换至相机归一化坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt,K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt,K));
  }
  
  cv::Mat pts_4d;
  cv::triangulatePoints(T1,T2,pts_1,pts_2,pts_4d);//https://blog.csdn.net/qq_41253960/article/details/123895928?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2-123895928-blog-123439617.pc_relevant_multi_platform_whitelistv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2-123895928-blog-123439617.pc_relevant_multi_platform_whitelistv2&utm_relevant_index=5
  // 输入：
  // T1: 3*4的cv::Mat ,就是第一张图片的 R和t组合一起的(这里以第一张图片为世界坐标系)
  // T2: 3*4的cv::Mat ,就是第二张图片的 R和t组合一起的
  // pts_1: vector<cv::Point2f> 前一张图片的匹配点(归一化坐标下)
  // pts_2: vector<cv::Point2f> 后一张图片的匹配点，肯定和上一个size()一样
  // 输出：
  // pts_4d: cv::Mat类型
  // 会输出4行，points1.size()列的Mat,一列代表一个点，把每一列的点都除以第4行的数，就是归一化的三维坐标了

  // 转换成非齐次坐标
  for (size_t i = 0; i < pts_4d.cols; i++)
  {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3,0);//归一化
    cv::Point3d p(x.at<float>(0,0),
                  x.at<float>(1,0),
                  x.at<float>(2,0));
    
    points.push_back(p);
  }
} 




cv::Point2f pixel2cam(const cv::Point2d &p ,const cv::Mat &K)
{
  return(cv::Point2f((p.x - K.at<double>(0,2)) / K.at<double>(0,0),
                     (p.y - K.at<double>(1,2)) / K.at<double>(1,1)));
}
