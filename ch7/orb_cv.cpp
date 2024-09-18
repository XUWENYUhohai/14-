#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

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
  cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);//CV_LOAD_IMAGE_COLOR = 1 ,返回彩色图像
  cv::Mat img_2 = cv::imread(argv[2],1);
  assert(img_1.data != nullptr && img_2.data != nullptr);



  //第一种

  //-- 初始化
  std::vector<cv::KeyPoint> keypoints_1,keypoints_2;
  cv::Mat descriptors_1,descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); //特征检测器
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
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "extract ORB cost = " << time_used.count() <<  " seconds. " << endl;
  cout << "descriptors_1 = " << descriptors_1.size() << endl;

  cv::Mat outimg1;
  //https://blog.csdn.net/leonardohaig/article/details/81289648
  //Scalar::all(-1)表示随机颜色
  //cv::DrawMatchesFlags特征点绘制模式，DEFAULT：只显示特征点的坐标
  cv::drawKeypoints(img_1,keypoints_1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);//Scalar就是一个可以用来存放4个double数值的数组(O'Reilly的书上写的是4个整型成员）；一般用来存放像素值（不一定是灰度值哦）的，最多可以存放4个通道的。
  cv::imshow("ORB features", outimg1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<cv::DMatch> matches;//Dmatch对象保存的是匹配成功的结果，当然这个匹配结果里面包含了不少的误匹配
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1,descriptors_2,matches);
  //FlannMatcher->match(descriptors_1,descriptors_2,matches);

  vector<vector<cv::DMatch>> KnnMatch;//返回的这俩个DMatch数据类型是俩个与原图像特征点最接近的俩个特征点
  matcher->knnMatch(descriptors_1,descriptors_2,KnnMatch,2);//KNN匹配，2为KNN中的k

  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(),matches.end(),[](const cv::DMatch &m1 , const cv::DMatch &m2){return m1.distance < m2.distance;});//https://blog.csdn.net/liyunlong19870123/article/details/113987617
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("--Max dist : %f \n",max_dist);
  printf("--Min dist : %f \n",min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<cv::DMatch> good_matches;//https://blog.csdn.net/ouyangandy/article/details/88997102
  for (size_t i = 0; i < matches.size(); i++)
  //for (size_t i = 0; i < descriptors_1.rows; i++)
  {
    if (matches[i].distance <= max(2 * min_dist,30.0))
    {
      good_matches.push_back(matches[i]);
    }
    
  }

  for (size_t i = 0; i < good_matches.size(); i++)
  {
          cout << keypoints_1[good_matches[i].queryIdx].pt << endl;
  }
  

  //-- 第五步:绘制匹配结果
  cv::Mat img_match,img_goodmatch; 
  cv::drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);//https://blog.csdn.net/two_ye/article/details/100576029
  cv::drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_goodmatch);
  cv::imshow("all matches", img_match);
  cv::imshow("good matches", img_goodmatch);
  cv::waitKey();


  // //第二种

  // //1.ORB检测
	// cv::Ptr<cv::ORB> orb = cv::ORB::create();
	// //2.生成描述子  
	// vector<cv::KeyPoint> kp1, kp2;
	// cv::Mat des1, des2;
	// orb->detectAndCompute(img_1, cv::Mat(), kp1, des1);//https://blog.csdn.net/caogps/article/details/107850859
	// orb->detectAndCompute(img_2, cv::Mat(), kp2, des2);//第二个参数cv::Mat()为掩模：https://blog.csdn.net/bitcarmanlee/article/details/79132017?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-79132017-blog-90696292.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-79132017-blog-90696292.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=1
 
	// //3.两个描述子匹配，两个mat 做for循环 只不过在这用BFMatcher
	// //3.1 定义一个类对象
	// //3.2 把两个mat放进去 输出的是matches 汉明距离作为距离度量
	// //3.3 matches应该是一个容器，里面是mat
	// //3.4 排序取前10个。
	// cv::BFMatcher bf(cv::NORM_HAMMING, true);
	// vector <cv::DMatch> matches;
	// bf.match(des1, des2, matches, cv::Mat());//值放入matches，第四个参数：//指定输入查询和描述符的列表矩阵之间的允许匹配的掩码

  // vector<vector<cv::DMatch>> KnnMatch;//返回的这俩个DMatch数据类型是俩个与原图像特征点最接近的俩个特征点
  // bf.knnMatch(des1, des2,KnnMatch,2);//KNN匹配，2为KNN中的k

	// sort(matches.begin(), matches.end());//排序
	// matches = vector<cv::DMatch>(matches.begin(), matches.begin() + 10);//重新赋值取前10个
	// vector<char> match_mask(matches.size(), 1);//新建一个容器 个数应该是10个

  // cv::Mat img_match;
  // cv::drawMatches(img_1,kp1,img_2,kp2,matches,img_match);//https://blog.csdn.net/two_ye/article/details/100576029
  // cv::imshow("all matches", img_match);
  // cv::waitKey();

  

  return 0;
}
