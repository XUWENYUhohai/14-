#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/format.hpp>// for formating strings

// https://blog.csdn.net/weixin_46098577/article/details/110951471
#include <pcl/point_types.h>//点类型定义,包含了各种点云数据的声明
#include <pcl/io/pcd_io.h>//PCD文件的读写
#include <pcl/filters/voxel_grid.h>//基于体素网格化的滤波
// #include <pcl/visualization/pcl_visualizer.h>//PCLVisualizer可视化类是PCL中功能最全的可视化类，与CloudViewer可视化类相比，PCLVisualizer使用起来更为复杂，但该类具有更全面的功能，如显示法线、绘制多种形状和多个视口。
#include <pcl/filters/statistical_outlier_removal.h>////统计离群点

using namespace std;

int main(int argc, char *argv[])
{
    vector<cv::Mat> colorImg , depthImg;// 彩色图和深度图
    vector<Eigen::Isometry3d> poses;// 相机位姿

    ifstream fin("./ch12/dense_RGBD/data/pose.txt");
    if(!fin)
    {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (size_t i = 0; i < 5; i++)
    {
        boost::format fmt("./ch12/dense_RGBD/data/%s/%d.%s");//图像文件格式
        // boost::format fmt("./ch12/dense_RGBD/data/%1%/%2%.%3%");
        colorImg.push_back(cv::imread((fmt %"color" %(i + 1) %"png").str()));
        depthImg.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(),-1));// 使用-1读取原始图像

        double data[7];
        for (size_t i = 0; i < 7; i++)
        {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6],data[3],data[4],data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0],data[1],data[2]));
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为点云..." << endl;

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // 新建一个点云
    PointCloud::Ptr pointcloud(new PointCloud);

    for (size_t i = 0; i < 5; i++)
    {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImg[i];
        cv::Mat depth = depthImg[i];
        Eigen::Isometry3d T = poses[i];
        for (size_t v = 0; v < color.rows; v++)
        {
            for (size_t u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];// 深度值
                if(d == 0) continue;// 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];//data返回uchar（8位：0-255）
                // p.b = color.at<cv::Vec3b>(v,u)[0];//b代表uchar
                // p.b = color.ptr(v)[u * color.channels()];//这个ptr不用<>,ptr默认返回uchar
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];                

                current->points.push_back(p);
            }
        }
        
        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;//https://blog.csdn.net/romagnoli13/article/details/119540501
        statistical_filter.setMeanK(50);//设置在进行统计时考虑的临近点个数(k个领域）
        statistical_filter.setStddevMulThresh(1.0);////设置判断是否为离群点的阀值，即std_mul_，用来倍乘标准差:作为离群点判断的阈值distance_threshold = mean(均值) + std_mul_ * stddev（标准差）
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        // (* pointcloud) += *tmp;
        * pointcloud += *tmp;
    }

    pointcloud->is_dense = false;//指点云中的所有数据都是有限的（true），还是其中的一些点不是有限的，它们的XYZ值可能包含inf/NaN 这样的值（false）。
    cout << "点云共有" << pointcloud->size() << "个点." << endl;

    // voxel filter     https://blog.csdn.net/Small_Munich/article/details/108348164
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.030;
    voxel_filter.setLeafSize(resolution,resolution,resolution); // resolution, // 设置每个体素的大小, leaf_size分别为lx ly lz的长 宽 高
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointcloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointcloud);

    cout << "滤波之后，点云共有" << pointcloud->size() << "个点." << endl;

    pcl::io::savePCDFileBinary("./ch12/map.pcd",*pointcloud);//https://anyaojun.github.io/2018/12/04/PCL%E7%B3%BB%E5%88%971%E2%80%94%E2%80%94%E8%AF%BB%E5%85%A5%E5%92%8C%E8%BE%93%E5%87%BApcd%E6%A0%BC%E5%BC%8F%E7%82%B9%E4%BA%91%E6%96%87%E4%BB%B6/
    // pcl::io::savePCDFile("./ch12/map.pcd",*pointcloud);//默认ASCII方式保存
    // pcl::io::savePCDFileASCII("./ch12/map.pcd",*pointcloud);//ASCII方式保存

    return 0;
}
