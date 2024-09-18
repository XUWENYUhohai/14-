#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "./ch8/LK1.png";
string file_2 = "./ch8/LK2.png";

// Optical flow tracker and interface
class OpticalFlowTracker
{
private:
    const cv::Mat &img1;//所以cv：：Mat里存放的像素都是像素坐标系下的整数坐标的
    const cv::Mat &img2;
    const vector<cv::KeyPoint> &kp1;
    vector<cv::KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true , has_initial = false;

public:
    OpticalFlowTracker(const cv::Mat &img1_, 
                       const cv::Mat &img2_ , 
                       const vector<cv::KeyPoint> &kp1_,
                       vector<cv::KeyPoint> &kp2_, 
                       vector<bool> &success_,
                       bool inverse_ = true , bool has_initial_ = false) : 
                       img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), 
                       has_initial(has_initial_)  {}

    void calculateOpticalFlow(const cv::Range &range);// //计算指定range范围内的特征点的光流
    //cv::range,得到一个整数序列，区间为[start,end)
};


/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false);



/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(const cv::Mat &img1,
                           const cv::Mat &img2,
                           const vector<cv::KeyPoint> &kp1,
                           vector<cv::KeyPoint> &kp2,
                           vector<bool> &success,
                           bool inverse = false);


/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
//使用双线性内插法来估计一个点的像素：https://www.cnblogs.com/yibeimingyue/p/11485732.html
inline float GetPixelValue(const cv::Mat &img , float x , float y)
{
// cout << "x:" << x << " y:" << y << endl;
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
// cout << "cols:" << img.cols << " rows:" << img.rows << endl;
// cout << "x:" << x << " y:" << y << endl;

    float xx = x - floor(x);//floor向下取整
    float yy = y - floor(y);//像素坐标有可能不是整数，不是说（x，y）坐标就对应某一像素点，小数的产生由于数学计算，因为坐标是连续的并非离散，https://blog.csdn.net/jiangqixing0728/article/details/124611910
    // int x_a1 = std::min(img.cols - 1 , int(x)+1);
    // int y_a1 = std::min(img.rows - 1 , int(y)+1);
    int x_a1 = int(x)+1;
    int y_a1 = int(y)+1;
// cout << "xx:" << xx << " yy:" << yy << endl;
// cout << "x_a1:" << x_a1 << " y_a1:" << y_a1 << endl;

    return(1 - xx) * (1 - yy) * img.at<uchar>(y,x) + //这里（x，y）已经是整数了，因为at里面存放的是int型,向下取整
    xx * (1 - yy) * img.at<uchar>(y,x_a1) + 
    (1 - xx) * yy * img.at<uchar>(y_a1,x) + 
    xx * yy * img.at<uchar>(y_a1,x_a1);
}



int main(int argc, char *argv[])
{
    // images, note they are CV_8UC1, not CV_8UC3
    cv::Mat img1 = cv::imread(file_1,0);//以灰度图读取
    cv::Mat img2 = cv::imread(file_2,0);

    // key points, using GFTT here.
    vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500,0.01,20);//https://blog.csdn.net/bashendixie5/article/details/125333554
    // cv::Ptr<cv::Feature2D> detector = cv::GFTTDetector::create(500,0.01,20);
    // cv::Ptr<cv::FeatureDetector> detector = cv::GFTTDetector::create(500,0.01,20);
    detector->detect(img1,kp1);//GFTT特征点只支持提取特征点，不支持计算描述子

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<cv::KeyPoint> kp2_single;
    vector<bool> success_single;

    OpticalFlowSingleLevel(img1,img2,kp1,kp2_single,success_single);


    // then test multi-level LK
    vector<cv::KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1,img2,kp1,kp2_multi,success_multi,true);



    // use opencv's flow for validation
    vector<cv::Point2f> pt1,pt2;
    for (auto &kp : kp1)
    {
        pt1.push_back(kp.pt);
    }

    vector<uchar> status;
    vector<float> error;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1,img2,pt1,pt2,status,error);//https://blog.csdn.net/codedoctor/article/details/79175683     
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used  = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;


    // plot the differences of those functions
    cv::Mat img2_cv;
    //https://blog.csdn.net/weixin_51105360/article/details/113941015
    cv::cvtColor(img2,img2_cv,CV_GRAY2BGR);//cvtcolor()函数是一个颜色空间转换函数,,cv::COLOR_GRAY2BGR:从灰度空间转换到BGR颜色空间
    for (size_t i = 0; i < pt2.size(); i++)
    {
        if(status[i])
        {
            cv::circle(img2_cv,pt2[i],2,cv::Scalar(0,250,0),2);
            cv::line(img2_cv,pt1[i],pt2[i],cv::Scalar(0,250,0),2);
        }
    }

    cv::Mat img2_single;
    cv::cvtColor(img2,img2_single,CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++)
    {
        if(success_single[i])
        {
            cv::circle(img2_single,kp2_single[i].pt,2,cv::Scalar(0,250,0),2);
            cv::line(img2_single,kp1[i].pt,kp2_single[i].pt,cv::Scalar(0,250,0));
        }
    }

    cv::Mat img2_multi;
    cv::cvtColor(img2,img2_multi,CV_GRAY2BGR);
    for (size_t i = 0; i < kp2_multi.size(); i++)
    {
        if(success_multi[i])
        {
            cv::circle(img2_multi,kp2_multi[i].pt,2,cv::Scalar(0,255,0),2);
            cv::line(img2_multi,kp1[i].pt,kp2_multi[i].pt,cv::Scalar(0,255,0));
        }
    }
    


    cv::imshow("tracked by opencv",img2_cv);
    cv::imshow("tracked by single",img2_single);
    cv::imshow("tracked multi level",img2_multi);
    cv::waitKey();

    return 0;
}




void OpticalFlowSingleLevel(const cv::Mat &img1 , const cv::Mat &img2,
                            const vector<cv::KeyPoint> &kp1 , vector<cv::KeyPoint> &kp2,
                            vector<bool> &success , bool inverse , bool has_initial)
    {
        kp2.resize(kp1.size());
        success.resize(kp1.size());

        OpticalFlowTracker tracker(img1,img2,kp1,kp2,success,inverse,has_initial);
        //Range表示一个范围，即并行计算哪些需要追踪的点，kp1中存放的就是随机选的追踪点，bind是一个绑定函数，表示调用OpticalFlowTracker的实例tracker中的calculateOpticalFlow()，即计算光流。
        //std::placeholders::_1是占位符，表示传入的参数是tracker.calculateOpticalFlow()的第一个参数，此处Range(0, kp1.size())为传入的参数。
        cv::parallel_for_(cv::Range(0,kp1.size()),std::bind(&OpticalFlowTracker::calculateOpticalFlow,&tracker,placeholders::_1));//https://zhuanlan.zhihu.com/p/352268773
        //cv::parallel_for_是opencv封装的一个多线程接口，利用这个接口可以方便实现多线程，不用考虑底层细节，cv::Range表示要执行的操作总数
    }




void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range)
{
       // cout << img2.size() <<endl;

    // parameters
    int half_path_size = 4;//窗口的大小为 8 * 8
    int iteration = 10;//每一个角点迭代 10 次
    for (size_t i = range.start; i < range.end; i++)
    {
        auto kp = kp1[i];
        double dx = 0 , dy = 0;// dx,dy need to be estimated
        if(has_initial)//多层光流使用
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        } 

        double cost = 0 , lastcost = 0;
        bool succ = true;// indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();// hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();// bias
        Eigen::Vector2d J; // jacobian

        for (int iter = 0; iter < iteration; iter++)
        {
            if(inverse == false)
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else
            {//反向光流，H矩阵不变，只计算一次，每次迭代计算残差b
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_path_size; x < half_path_size; x++)
            {
                for (int y = -half_path_size; y < half_path_size; y++)
                {
                    double error = GetPixelValue(img1,kp.pt.x + x,kp.pt.y + y) - 
                                   GetPixelValue(img2,kp.pt.x + x + dx,kp.pt.y + y + dy);

                    if(inverse == false)
                    {//https://zhuanlan.zhihu.com/p/352268773,图像梯度：中值差分：Ix = （I（x+1，y）-I（x-1，y））/2 ， Iy = （I（x，y+1）-I（x，y-1））/2
                        J = - 1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2,kp.pt.x + x + dx + 1,kp.pt.y + y + dy) - 
                                   GetPixelValue(img2,kp.pt.x + x + dx - 1,kp.pt.y + y + dy)) ,  
                            0.5 * (GetPixelValue(img2,kp.pt.x + x + dx,kp.pt.y + y + dy + 1) - 
                                   GetPixelValue(img2,kp.pt.x + x + dx,kp.pt.y + y + dy - 1)));        
                    }
                    else if(iter == 0)
                    {// //反向光流，计算第一张图像的雅可比矩阵，10次迭代只计算一次          https://blog.csdn.net/level_code/article/details/123459188
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );//这里的负号应该对应上面error，因为反向光流的error为GetPixelValue(img2,kp.pt.x + x + dx,kp.pt.y + y + dy) - GetPixelValue(img1,kp.pt.x + x,kp.pt.y + y)
                    }

                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if(inverse == false || iter == 0)
                    {
                        // also update H
                        H += J * J.transpose();
                    }        
                }
            }
            
            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if(std::isnan(update[0]))
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if(iter > 0 && cost > lastcost)
            {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastcost = cost;
            succ = true;

            if(update.norm() < 1e-2)
            {
                //收敛
                break;
            }
        }
        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx,dy);
    }
}



void OpticalFlowMultiLevel(const cv::Mat &img1,const cv::Mat &img2,const vector<cv::KeyPoint> &kp1,vector<cv::KeyPoint> &kp2,vector<bool> &success,bool inverse)
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0,0.5,0.25,0.125};

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<cv::Mat> pyr1,pyr2;//image pyramids
    for (size_t i = 0; i < pyramids; i++)
    {
        if(i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr,img2_pyr;
            cv::resize(pyr1[i-1],img1_pyr,cv::Size(pyr1[i-1].cols * pyramid_scale , pyr1[i-1].rows * pyramid_scale));//https://blog.csdn.net/xidaoliang/article/details/86504720
            cv::resize(pyr2[i-1],img2_pyr,cv::Size(pyr2[i-1].cols * pyramid_scale , pyr2[i-1].rows * pyramid_scale));//缩放图像
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "build pyramid time: " << time_used.count() << endl;


    // coarse-to-fine LK tracking in pyramids
    vector<cv::KeyPoint> kp1_pyr,kp2_pyr;
    for (auto &kp : kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }
    
    for (int level = pyramids -1 ; level >= 0; level--)//这里不能是size_t，因为其无符号
    {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level],pyr2[level],kp1_pyr,kp2_pyr,success,inverse,true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if(level > 0 )
        {
            for (auto &kp : kp1_pyr)
            {
                kp.pt /= pyramid_scale;
            }
            
            for (auto &kp : kp2_pyr)
            {
                kp.pt /= pyramid_scale;
            }
        }
    }
    
    for (auto &kp : kp2_pyr)
    {
        kp2.push_back(kp);
    }
}