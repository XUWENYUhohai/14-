#include <pcl/point_cloud.h>//点云类定义
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>//KD-Tree搜索
// #include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>//最小二乘平滑处理
#include <pcl/surface/gp3.h>//贪婪投影三角化算法
// #include <pcl/surface/impl/mls.hpp>

//typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;//PointXYZRGBNormal存储XYZ数据和RGB颜色的point结构体，并且包括曲面法线和曲率，- float x, y, z, rgb, normal[3], curvature
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

SurfelCloudPtr reconstructSurface(const PointCloudPtr &input , float radius , int polynomial_order)//polynormial多项式
{
    //https://blog.csdn.net/u012337034/article/details/37534869
    pcl::MovingLeastSquares<PointT,SurfelT> mls;//PointT输入，SurfelT输出，利用移动最小二乘法完成滤波
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);//创建用于最近邻搜索的KD-Tree
    mls.setSearchMethod(tree);//使用kdTree加速搜索
    mls.setSearchRadius(radius);//确定搜索的半径。也就是说在这个半径里进行表面映射和曲面拟合。从实验结果可知：半径越小拟合后曲面的失真度越小，反之有可能出现过拟合的现象。(单位m)
    mls.setComputeNormals(true);//进行法线估计，设置在最小二乘计算中是否需要存储计算的法线
    mls.setSqrGaussParam(radius * radius);//设置用于基于距离的邻居加权的参数（搜索半径的平方通常效果最佳）。
    // https://blog.csdn.net/qq_45006390/article/details/119235490
    mls.setPolynomialFit(polynomial_order > 1);//对于法线的估计是有多项式还是仅仅依靠切线,  // 设置为false可以 加速 smooth,可以通过不需要多项式拟合来加快平滑速度，设置为ture时则在整个算法运行时采用多项式拟合来提高精度
    mls.setPolynomialOrder(polynomial_order);//几阶多项式拟合，这个阶数在构造函数里默认是2
    mls.setInputCloud(input);
    SurfelCloudPtr output(new SurfelCloud);
    mls.process(*output);//进行曲面重建
    return output;
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels)
{
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);
    tree->setInputCloud(surfels);//用surfels构造tree对象,好像不用也行

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3; // 定义三角化对象
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);// //存储最终三角化的网络模型

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);//该函数设置了三角化后得到的每个三角形的最大可能边长

    // Set typical values for the parameters
    gp3.setMu(2.5);//规定了被样本点搜索其邻近点的最远距离,mu是个加权因子，对于每个参考点，其映射所选球的半径由mu与离参考点最近点的距离乘积所决定，这样就很好解决了点云密度不均匀的问题，mu一般取值为2.5-3。
    gp3.setMaximumNearestNeighbors(100);//可搜索的邻域个数(临近点阈值设定)。一般为80-100。
    gp3.setMaximumSurfaceAngle(M_PI / 4);// 45 degrees, 两点的法向量角度差大于此值，这两点将不会连接成三角形，这个就恰恰和点云局部平滑的约束呼应，如果一个点是尖锐点那么它将不会和周围的点组成三角形，其实这个也有效的滤掉了一些离群点。这个值一般为45度。
    gp3.setMinimumAngle(M_PI / 18);// 10 degrees,三角形最小角度的阈值。
    gp3.setMaximumAngle(2 * M_PI / 3);// 120 degrees,三角形最大角度的阈值。
    gp3.setNormalConsistency(true);//输入的法向量是否连续变化的。 设置该参数为true保证法线朝向一致，设置为false的话不会进行法线一致性检查,这个一般来讲是false，除非输入的点云是全局光滑的（比如说一个球）。

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}

int main(int argc, char *argv[])
{
    // Load the points
    PointCloudPtr cloud(new PointCloud);
    if (argc == 0 || pcl::io::loadPCDFile(argv[1],*cloud))
    {
        cout << "failed to load point cloud!";
        return 1;
    }
    
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05 , polynomial_order = 2;
    auto surfels = reconstructSurface(cloud,mls_radius,polynomial_order);
    cout << "surfels = " << surfels->points.size() << endl;

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    cout << "display mesh ... " << endl;
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh,"mesh frame");
    vis.addPolygonMesh(*mesh,"mesh");
    vis.resetCamera();
    vis.spin();
    return 0;
}
