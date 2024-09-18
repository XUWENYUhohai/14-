#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>//貌似可以代替部分Eigen引用
#include <boost/format.hpp>// for formating strings字符串格式化
//https://blog.csdn.net/u014779536/article/details/116395236，https://wenku.baidu.com/view/2ea45b11edfdc8d376eeaeaad1f34693daef10f4.html
#include <mutex>

using namespace std;

typedef vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "./ch8/left.png";
string disparity_file = "./ch8/disparity.png";
boost::format fmt_others("./ch8/%06d.png"); // other files
// %06d 整数输出，宽度是6位，不足左边补数字0
//https://blog.csdn.net/Sunnyside_/article/details/117464078?utm_term=c%E8%AF%AD%E8%A8%80%2506d&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-0-117464078-null-null&spm=3001.4430

// useful typedefs
typedef Eigen::Matrix<double,6,6> Matrix6d;//H
typedef Eigen::Matrix<double,2,6> Matrix26d;//J^T,e的导数是2X6的
typedef Eigen::Matrix<double,6,1> Vector6d;//李代数

/// class for accumulator jacobians in parallel
class JacobianAccumulator
{
public:
    JacobianAccumulator(const cv::Mat &img1_,
                        const cv::Mat &img2_,
                        const VecVector2d &px_ref_,
                        const vector<double> depth_ref_,
                        Sophus::SE3d &T21_) : 
                        img1(img1_),img2(img2_),px_ref(px_ref_),depth_ref(depth_ref_),T21(T21_)
    {
        projection = VecVector2d(px_ref.size(),Eigen::Vector2d::Zero());
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const {return H;}

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }//图2的像素坐标

    /// reset h, b, cost to zero
    void reset() 
    {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection;// projected points


//线程：https://www.bilibili.com/read/cv3834364?spm_id_from=333.999.0.0
//互斥量：https://www.bilibili.com/read/cv3834615/
    std::mutex hessian_mutex;//多线程协同工作，往往要共享某一块或某几块内存区，由于数据写入的最后一步是刷新内存区，而各线程执行细节不完全可知可控，容易造成数据损失，为避免这种情况，使用互斥量保证某一线程使用该内存区时，其他线程不得访问，即独占
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};



/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(const cv::Mat &img1,
                                    const cv::Mat &img2,
                                    const VecVector2d &px_ref,
                                    const vector<double> depth_ref,
                                    Sophus::SE3d &T21);



void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);



// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img,float x,float y)
{
    // boundary check
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= img.cols) x = img.cols - 1;
    if(y >= img.rows) y = img.rows - 1;
    uchar * data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float((1 - xx) * (1 - yy) * data[0] + xx * (1 -yy) * data[1] + (1 -xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
    // return(1 - xx) * (1 - yy) * img.at<uchar>(y,x) + xx * (1 - yy) * img.at<uchar>(y,x_a1) + (1 - xx) * yy * img.at<uchar>(y_a1,x) + xx * yy * img.at<uchar>(y_a1,x_a1);
}



int main(int argc, char *argv[])
{
    cv::Mat left_img = cv::imread(left_file,0);
    cv::Mat disparity_img = cv::imread(disparity_file,0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int npoint = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < npoint; i++)
    {   //uniform：区间[boarder,left_img.cols - boarder）均匀分布
        int x = rng.uniform(boarder,left_img.cols - boarder);// don't pick pixels close to boarder
        int y = rng.uniform(boarder,left_img.rows - boarder);
        int disparity = disparity_img.at<uchar>(y,x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x,y));
    }


    //测试用关键点而不是随机点作为3d特征点
    // vector<cv::KeyPoint> kp;
    // cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500,0.01,20);
    // detector->detect(left_img,kp);
    // for (size_t i = 0; i < kp.size(); i++)
    // {
    //     int disparity = disparity_img.at<uchar>(kp[i].pt.y,kp[i].pt.x);
    //     double depth = fx * baseline / disparity;
    //     depth_ref.push_back(depth);
    //     pixels_ref.push_back(Eigen::Vector2d(kp[i].pt.x,kp[i].pt.y));
    // }
    
    
    
    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++)// 1~10
    {
        cout << i <<endl;

        cv::Mat img = cv::imread((fmt_others % i).str(),0);//.str()返回string类型
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    
    return 0;
}


void DirectPoseEstimationSingleLayer(const cv::Mat &img1,const cv::Mat &img2,const VecVector2d &px_ref,const vector<double> depth_ref,Sophus::SE3d &T21)
{
    const int iterations = 10;
    double cost = 0,lastcost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1,img2,px_ref,depth_ref,T21);
    

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0,px_ref.size()),std::bind(&JacobianAccumulator::accumulate_jacobian,&jaco_accu,std::placeholders::_1));

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        
        cost = jaco_accu.cost_func();

        if(std::isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }

        if(iter > 0 && cost > lastcost)
        {
            cout << "cost increased: " << cost << ", " << lastcost << endl;
            break;
        }

        if(update.norm() < 1e-3)
        {
            // converge
            break;
        }


        T21 = Sophus::SE3d::exp(update) * T21;
        lastcost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }
    
    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img_show;
    cv::cvtColor(img2,img_show,CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (int i = 0; i < px_ref.size(); i++)
    {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if(p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img_show,cv::Point2f(p_cur[0],p_cur[1]),2,cv::Scalar(0,255,0),2);
            cv::line(img_show,cv::Point2f(p_ref[0],p_ref[1]),cv::Point2f(p_cur[0],p_cur[1]),cv::Scalar(0,255,0));
        }
    }
    cv::imshow("current",img_show);
    cv::waitKey();
}



void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
{
    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++)
    {
        // compute the projection in the second image
        Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx , (px_ref[i][1] - cy) / fy , 1);//图1坐标系下的3d点
        Eigen::Vector3d point_cur = T21 * point_ref;//图2坐标系下3d点
        if(point_cur[2] < 0)
        {
            continue;// depth invalid
        }

        float u = fx * point_cur[0] / point_cur[2] + cx , v = fy * point_cur[1] / point_cur[2] + cy;

        if(u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size || v > img2.rows - half_patch_size) continue;

        projection[i] = Eigen::Vector2d(u,v);
        double X = point_cur[0] , Y = point_cur[1] , Z = point_cur[2] , Z2 = Z * Z , Z_inv = 1.0 / Z , Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
        {
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                double error = GetPixelValue(img1,px_ref[i][0] + x , px_ref[i][1] + y) - GetPixelValue(img2,u + x,v + y);

                Matrix26d J_pixel_xi;//2X6 李代数右扰动的导数
                Eigen::Vector2d J_img_pixel;
                
                J_pixel_xi(0,0) = fx * Z_inv;
                J_pixel_xi(0,1) = 0;
                J_pixel_xi(0,2) = -fx * X * Z2_inv;
                J_pixel_xi(0,3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0,4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0,5) = -fx * Y * Z_inv;

                J_pixel_xi(1,0) = 0;
                J_pixel_xi(1,1) = fy * Z_inv;
                J_pixel_xi(1,2) = -fy * Y * Z2_inv;
                J_pixel_xi(1,3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1,4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1,5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(0.5 * (GetPixelValue(img2,u + x + 1,v + y) - GetPixelValue(img2,u + x -1,v + y)) , 0.5 * (GetPixelValue(img2,u + x,v + y + 1) - GetPixelValue(img2,u + x,v + y - 1)));

                //使用img1的梯度作为第二个图像的梯度
                // J_img_pixel = Eigen::Vector2d(0.5 * (GetPixelValue(img1,px_ref[i](0) + x + 1,px_ref[i](1) + y) - GetPixelValue(img1,px_ref[i](0) + x - 1,px_ref[i](1) + y)) , 0.5 * (GetPixelValue(img1,px_ref[i](0) + x,px_ref[i](1) + y + 1) - GetPixelValue(img1,px_ref[i](0) + x,px_ref[i](1) + y - 1)));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();//J.transpose()是e的导数  ， 1X6
                bias += -error * J;
                cost_tmp += error * error;
            }
        }
    }

    if(cnt_good)
        {
            // set hessian, bias and cost
            //https://blog.csdn.net/wangxu696200/article/details/122775502                https://blog.csdn.net/YMWM_/article/details/117514010
            unique_lock<mutex> lck(hessian_mutex);//加入一个线程锁，使线程执行有先后顺序(因为H和b矩阵累积需要有先后顺序)，这个对象生命周期结束后自动解锁
            H += hessian;
            b += bias;
            cost += cost_tmp / cnt_good;
        }
}


//多层改变了像素层的分辨率但实际的3d点不变
void DirectPoseEstimationMultiLayer(const cv::Mat &img1,const cv::Mat &img2,const VecVector2d &px_ref,const vector<double> depth_ref,Sophus::SE3d &T21)
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1,pyr2;
    for (size_t i = 0; i < pyramids; i++)
    {
        if(i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr , img2_pyr;
            cv::resize(pyr1[i-1],img1_pyr,cv::Size(pyr1[i-1].cols * pyramid_scale , pyr1[i-1].rows * pyramid_scale));
            cv::resize(pyr2[i-1],img2_pyr,cv::Size(pyr2[i-1].cols * pyramid_scale , pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (size_t level = pyramids -1 ; level >= 0; level--)
    {
        VecVector2d px_ref_pyr;// set the keypoints in this pyramid leve
        for (auto &px : px_ref)
        {
            px_ref_pyr.push_back(px * scales[level]);
        }
        
        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];

        DirectPoseEstimationSingleLayer(pyr1[level],pyr2[level],px_ref_pyr,depth_ref,T21);
    }
    
    
}