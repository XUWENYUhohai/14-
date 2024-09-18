#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
// using namespace cv;

double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

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
  cv::Mat img_2 = cv::imread(argv[2],1);//CV_LOAD_IMAGE_COLOR = 1 ,返回彩色图像

  cv::Mat img_3 = cv::imread(argv[1],0);
  cv::Mat img_4 = cv::imread(argv[2],0);

  assert(img_1.data != nullptr && img_2.data != nullptr);



  //第一种

  //-- 初始化
  std::vector<cv::KeyPoint> keypoints_1,keypoints_2, keypoints_3;
  cv::Mat descriptors_1,descriptors_2, descriptors_3;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(100); //特征检测器
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();//描述子提取
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");//描述子匹配器

  cv::Ptr<cv::FlannBasedMatcher> FlannMatcher = cv::FlannBasedMatcher::create();//FLANN
  

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1,keypoints_1);
  detector->detect(img_2,keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1,keypoints_1,descriptors_1);
  descriptor->compute(img_2,keypoints_2,descriptors_2);

  vector<cv::DMatch> matches;
  matcher->match(descriptors_1,descriptors_2,matches);

//   vector<vector<cv::DMatch>> KnnMatch;//返回的这俩个DMatch数据类型是俩个与原图像特征点最接近的俩个特征点
//   matcher->knnMatch(descriptors_1,descriptors_2,KnnMatch, 1);//KNN匹配，2为KNN中的k

  auto min_max = minmax_element(matches.begin(),matches.end(),[](const cv::DMatch &m1 , const cv::DMatch &m2){return m1.distance < m2.distance;});//https://blog.csdn.net/liyunlong19870123/article/details/113987617
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<cv::DMatch> good_matches;//https://blog.csdn.net/ouyangandy/article/details/88997102
  for (size_t i = 0; i < matches.size(); i++)
  //for (size_t i = 0; i < descriptors_1.rows; i++)
  {
    if (matches[i].distance <= max(2 * min_dist,80.0))
    {
      good_matches.push_back(matches[i]);
    }
    
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "match ORB cost = " << time_used.count() <<  " seconds. " << endl;
  cout << "matches = " << matches.size() << endl;
  cout << "good matches = " << good_matches.size() << endl;

  cv::Mat img_match1,img_match2; 
  cv::drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_match1);//https://blog.csdn.net/two_ye/article/details/100576029
  cv::imshow("img_match1", img_match1);
  cv::waitKey();



//GFTT
    vector<cv::Point2f> points1, points2;
    vector<uchar> status, status_reverse;
    vector<float> err;
    int sum = 0;
    t1 = chrono::steady_clock::now();

    // cv::goodFeaturesToTrack(img_1,points, 500, 0.01, 30, cv::Mat());
    cv::goodFeaturesToTrack(img_3,points1, 100, 0.01, 30);

    // cv::calcOpticalFlowPyrLK(img_3, img_4, points1, points2, status, err, cv::Size(21, 21), 1);
    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, cv::Size(21, 21), 3);




    for (size_t i = 0; i < status.size(); i++)
    {
        if(status[i])
        {
            sum++;
        }
    }

    cout << "status = " << sum << endl;


    vector<cv::Point2f> reverse_pts = points1;
    cv::calcOpticalFlowPyrLK(img_2, img_1, points2, reverse_pts, status_reverse, err, cv::Size(21, 21), 3);

    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "extract GFTT cost = " << time_used.count() <<  " seconds. " << endl;

    sum = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        // if(status[i] && status_reverse[i] && distance(points1[i], reverse_pts[i]) <= 10)
        if(status[i])
        {
            status[i] = 1;
            sum++;
        }
        else
        {
            status[i] = 0;
        }
    }

    cout << "good status = " << sum << endl;

    cv::hconcat(img_1, img_2, img_match2);

    for(size_t j = 0; j < points1.size(); j++)
    {
        cv::circle(img_match2, points1[j], 2, cv::Scalar(0,0,255), 2);

        if(status[j])
        { 
            cv::Point2f tmp = points2[j];
            tmp.x += img_1.cols;
            cv::circle(img_match2, tmp, 2, cv::Scalar(255,0,0), 2);
            cv::line(img_match2, points1[j], tmp, cv::Scalar(0,255,0), 1, 8, 0);
        }
    }


    cv::imshow("img_match2", img_match2);
    cv::waitKey();



//AKAZE
    
    // t1 = chrono::steady_clock::now();
    // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    // akaze->detectAndCompute(img_1, cv::Mat(), keypoints_3, descriptors_3);

    // t2 = chrono::steady_clock::now();
    // time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    // keypoints_3.resize(500);
    // cout << "extract AKAZE cost = " << time_used.count() <<  " seconds. " << endl;
    // cout << "keypoints_3 = " << keypoints_3.size() << endl;

    // cv::Mat outimg3;
    // cv::drawKeypoints(img_1,keypoints_3,outimg3,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);//Scalar就是一个可以用来存放4个double数值的数组(O'Reilly的书上写的是4个整型成员）；一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
    // cv::imshow("AKAZE features", outimg3);
    // cv::waitKey();

    cv::imwrite("../img_match1.png", img_match1);
    cv::imwrite("../img_match2.png", img_match2);


  return 0;
}
