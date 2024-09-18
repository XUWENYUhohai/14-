#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main(int argc, char *argv[])
{
    // read the image 
    cout<<"reading images... "<<endl;
    vector<cv::Mat> images;
    for (size_t i = 0; i < 10; i++)
    {
        string path = "./ch11/data/" + to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // cv::Ptr<cv::ORB> detector = cv::ORB::create();

    vector<cv::Mat> descriptors;
    for (cv::Mat & image : images)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image,cv::Mat(),keypoints,descriptor);
        descriptors.push_back(descriptor);
    }
    
    // create vocabulary
    cout<<"creating vocabulary ... "<<endl;
    //// BoW 默认构造函数，k=10，d=5，最多容纳10^5=100000个视觉单词
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);// 添加需要构造字典的ORB视觉特征描述子向量，10幅图像，每幅图像500个特征

    if(vocab.empty()) cout << "空" << endl;

    cout << "vocabulary info:" << vocab << endl;
    vocab.save("./ch11/vocabulary.yml.gz");//字典存储为一个压缩文件
    cout<<"done"<<endl;
    return 0;
}
