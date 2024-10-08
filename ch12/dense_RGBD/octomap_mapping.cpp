#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>// for octomap 

#include <Eigen/Geometry>
#include <boost/format.hpp>

int main(int argc, char *argv[])
{
    std::vector<cv::Mat> colorImgs , depthImgs;
    std::vector<Eigen::Isometry3d> poses;

    std::ifstream fin("./ch12/dense_RGBD/data/pose.txt");
    if(!fin)
    {
        std::cerr << "cannot find pose file" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < 5; i++)
    {
        boost::format fmt("./ch12/dense_RGBD/data/%s/%d.png");
        colorImgs.push_back(cv::imread((fmt %"color" %(i + 1)).str()));
        depthImgs.push_back(cv::imread((fmt %"depth" %(i + 1)).str(),-1));// 使用-1读取原始图像

        double data[7] = {0};
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

    std::cout << "正在将图像转换为 Octomap ..." << std::endl;

    // octomap tree 
    octomap::OcTree tree(0.01);// 参数为分辨率，单位（m）,经过N次划分后得到的小立方体的边长，我们称为地图的分辨率(resolution)

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << "转换图像中: " << i + 1 << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud; // the point cloud in octomap 

        for (size_t v = 0; v < color.rows; v++)
        {
            for (size_t u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if(d == 0) continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                // 将世界坐标系的点放入点云
                cloud.push_back(pointWorld[0],pointWorld[1],pointWorld[2]);
            }
        }
        
        // 将点云存入八叉树地图，给定传感器原点，这样可以计算投射线
        tree.insertPointCloud(cloud,octomap::point3d(T(0,3),T(1,3),T(2,3)));
    }
    
    // 更新中间节点的占据信息并写入磁盘
    tree.updateInnerOccupancy();
    std::cout << "saving octomap ... " << std::endl;
    tree.writeBinary("./ch12/octomap.bt");

    return 0;
}
