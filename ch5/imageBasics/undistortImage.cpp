#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>

using namespace std;

string image_file = "~/learn_slam/ch5/imageBasics/distorted.png";// 请确保路径正确

int main(int argc, char *argv[])
{
    // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;

    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;


    // cv::Mat image = cv::imread(image_file); // 图像是灰度图，CV_8UC1
    cv::Mat image = cv::imread(argv[1],0); // 图像是灰度图，CV_8UC1


    // 使用函数cv2.imread(filepath,flags)读入一副图片
    // filepath：要读入图片的完整路径
    // flags：读入图片的标志
    // cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可以直接写1
    // cv2.IMREAD_GRAYSCALE：读入灰度图片，可以直接写0
    // cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可以直接写-1
    // cv2.imread()读取图片后以多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引

    int rows = image.rows , cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows,cols,CV_8UC1);// 去畸变以后的图
    cv::Mat image_undistort2 = cv::Mat(rows,cols,CV_8UC1);

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
            double x = (u - cx) / fx ,y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows)
            {
                image_undistort.at<uchar>(v,u) = image.at<uchar>((int)v_distorted,(int)u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v,u) = 0;
            }
        }
    }

    // 画图去畸变后图像
    cv::imshow("distorted",image);
    cv::imshow("undistorted",image_undistort);
    // cv::imwrite("undistorted.png", image_undistort);
    cv::waitKey();//== cv::waitKey(0);



    //用cv::undistort()去畸变
    //     void cv::undistort	
    // (	    InputArray 	src,                        // 原始图像
    //         OutputArray 	dst,                    // 矫正图像
    //         InputArray 	cameraMatrix,               // 原相机内参矩阵
    //         InputArray 	distCoeffs,                 // 相机畸变参数
    //         InputArray 	newCameraMatrix = noArray() // 新相机内参矩阵
    // )	
    // cv::Mat K = ( cv::Mat_<double>(3,3) << 458.654,0.0,367.215,0.0,457.296,248.375,0.0,0.0,1.0 );//https://blog.csdn.net/u012058778/article/details/90764430
    // cv::Mat D = ( cv::Mat_<double>(5,1) << -0.28340811,0.07395907,0,0.00019359,1.76187114e-05 );//畸变项
    // cv::undistort(image,image_undistort2,K,D,K);
    // cv::imshow("distorted",image);
    // cv::imshow("undistort2",image_undistort2);
    // cv::waitKey();

    return 0;
}
