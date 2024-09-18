#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
// using namespace cv;

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }

  //-- 读取图像
//   cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);//CV_LOAD_IMAGE_COLOR = 1 ,返回彩色图像
  cv::Mat img_1 = cv::imread(argv[1],1);//CV_LOAD_IMAGE_COLOR = 1 ,返回彩色图像
  cv::Mat img_2 = cv::imread(argv[2],0);
  assert(img_1.data != nullptr && img_2.data != nullptr);



  //第一种

  //-- 初始化
  std::vector<cv::KeyPoint> keypoints_1,keypoints_2, keypoints_3;
  cv::Mat descriptors_1,descriptors_2, descriptors_3;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); //特征检测器
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();//描述子提取
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");//描述子匹配器

  cv::Ptr<cv::FlannBasedMatcher> FlannMatcher = cv::FlannBasedMatcher::create();//FLANN
  

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1,keypoints_1);
//   detector->detect(img_2,keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
//   descriptor->compute(img_1,keypoints_1,descriptors_1);
//   descriptor->compute(img_2,keypoints_2,descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "extract ORB cost = " << time_used.count() <<  " seconds. " << endl;
  cout << "keypoints_1 = " << keypoints_1.size() << endl;

  cv::Mat outimg1;
  //https://blog.csdn.net/leonardohaig/article/details/81289648
  //Scalar::all(-1)表示随机颜色
  //cv::DrawMatchesFlags特征点绘制模式，DEFAULT：只显示特征点的坐标
  cv::drawKeypoints(img_1,keypoints_1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);//Scalar就是一个可以用来存放4个double数值的数组(O'Reilly的书上写的是4个整型成员）；一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
  cv::imshow("ORB features", outimg1);
  cv::waitKey();

//GFTT
    vector<cv::Point2f> points;
    t1 = chrono::steady_clock::now();

    // cv::goodFeaturesToTrack(img_1,points, 500, 0.01, 30, cv::Mat());
    cv::goodFeaturesToTrack(img_2,points, 500, 0.01, 15);

    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "extract GFTT cost = " << time_used.count() <<  " seconds. " << endl;
    cout << "keypoints_2 = " << points.size() << endl;

    // for (int i = 0; i < points.size(); i++)
    // {
    //     keypoints_2.push_back(cv::KeyPoint(points[i],1.f));
    // }
    cv::KeyPoint::convert(points, keypoints_2);

    cv::Mat outimg2;
    cv::drawKeypoints(img_1,keypoints_2,outimg2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);//Scalar就是一个可以用来存放4个double数值的数组(O'Reilly的书上写的是4个整型成员）；一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
    cv::imshow("GFTT features", outimg2);
    cv::waitKey();



//AKAZE
    
    t1 = chrono::steady_clock::now();
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(img_1, cv::Mat(), keypoints_3, descriptors_3);

    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    keypoints_3.resize(500);
    cout << "extract AKAZE cost = " << time_used.count() <<  " seconds. " << endl;
    cout << "keypoints_3 = " << keypoints_3.size() << endl;

    cv::Mat outimg3;
    cv::drawKeypoints(img_1,keypoints_3,outimg3,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);//Scalar就是一个可以用来存放4个double数值的数组(O'Reilly的书上写的是4个整型成员）；一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
    cv::imshow("AKAZE features", outimg3);
    cv::waitKey();

    cv::imwrite("../outimg1.png", outimg1);
    cv::imwrite("../outimg2.png", outimg2);
    cv::imwrite("../outimg3.png", outimg3);

  return 0;
}
