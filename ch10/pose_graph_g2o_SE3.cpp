#include <iostream>
#include <fstream>
// #include <string>

#include <g2o/types/slam3d/types_slam3d.h>// g2o/types/slam3d/中的SE3表示位姿(顶点类型)
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;
/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }

    ifstream fin(argv[1]);

    if (!fin)
    {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block;
    typedef g2o::LinearSolverEigen<Block::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<Block>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int vertexCnt = 0,edgeCnt = 0;// 顶点和边的数量
    while (!fin.eof())
    {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
//read的原型， https://zhuanlan.zhihu.com/p/261891075?utm_source=cn.wiz.note
//bool curveVetex::read(std::istream &is)
// {
//     is >> _estimate[0] >> _estimate[1] >> _estimate[2] >> ......;
//     return true;
// }
            v->read(fin);//位姿顶点
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
            {
                v->setFixed(true);//要优化的变量设置为false
            }
        }
        else if(name == "EDGE_SE3:QUAT")
        {
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1,idx2;// 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0,optimizer.vertices()[idx1]);
            e->setVertex(1,optimizer.vertices()[idx2]);
            e->read(fin);//观测和信息矩阵右上角
            optimizer.addEdge(e);
        }

        if(!fin.good()) break;
//is_open()：文件是否正常打开
// bad()：读写过程中是否出错(操作对象没有打开，写入的设备没有空间)
// fail()：读写过程中是否出错(操作对象没有打开，写入的设备没有空间，格式错误--比如读入类型不匹配)
// eof()：读文件到达文件末尾，返回true
// good()：以上任何一个返回true，这个就返回false
// https://blog.csdn.net/weixin_42587961/article/details/86677869        
    }
    
    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save("./ch10/result.g2o");

    return 0;
}
