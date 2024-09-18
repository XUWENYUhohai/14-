#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
//unistd.h为Linux/Unix系统中内置头文件，包含了许多系统服务的函数原型，例如read函数、write函数和getpid函数等。
//其作用相当于windows操作系统的"windows.h"，是操作系统为用户提供的统一API接口，方便调用系统提供的一些服务。

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// path to trajectory file
string trajectory_file = "./ch3/examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d,Eigen::aligned_allocator<Isometry3d>>);//aligned_allocator:https://blog.csdn.net/chengde6896383/article/details/83183444

int main(int argc, char *argv[])
{
    vector<Isometry3d,Eigen::aligned_allocator<Isometry3d>> poses;
    ifstream fin(trajectory_file);
    if(!fin)
    {
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }

    while (!fin.eof())
    {
        double time,tx,ty,tz,qx,qy,qz,qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw,qx,qy,qz));//Twr.rotate(Quaternion(qw,qx,qy,qz));
        Twr.pretranslate(Vector3d(tx,ty,tz)); 
        poses.push_back(Twr);
    }

    cout << "read total " << poses.size() << " pose entries" << endl;

    // draw trajectory in pangolin
    DrawTrajectory(poses);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d,Eigen::aligned_allocator<Isometry3d>> poses)//https://blog.csdn.net/weixin_45929038/article/details/122904705
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer",1024,768);//新建一个窗口
    glEnable(GL_DEPTH_TEST);//glEnable 用于启用各种功能。功能由参数决定。GL_DEPTH_TEST:启用深度测试可以使得Pangolin构建的相机只展示镜头朝向一侧的像素信息，从而避免容易混淆的透视关系。根据坐标的远近自动隐藏被遮住的图形（材料）
    glEnable(GL_BLEND);//GL_BLEND:启用颜色混合。例如实现半透明效果
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);//颜色混合:OpenGL会把源颜色和目标颜色各自取出，并乘以一个系数（源颜色乘以的系数称为“源因子”，目标颜色乘以的系数称为“目标因子”），然后相加，这样就得到了新的颜色。
    //glBlendFunc有两个参数，前者表示源因子，后者表示目标因子。
    // GL_SRC_ALPHA                      0x0302//表示使用源颜色的alpha值来作为因子
    //GL_ONE_MINUS_SRC_ALPHA            0x0303//表示用1.0减去源颜色的alpha值来作为因子

    //构建观察相机对象
    pangolin::OpenGlRenderState s_cam(//定义投影和初始模型视图矩阵
        pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),//    projection_matrix：用于构建观察相机的内参系数：    w、h：相机的视野宽、高，fu、fv、u0、v0相机的内参，对应《视觉SLAM十四讲》中内参矩阵的fx、fy、cx、cy，zNear、zFar：相机的最近、最远视距
        //对应的是gluLookAt,摄像机初始位置,（相机视点的初始位置）参考点位置,up：相机自身哪一轴朝上放置，可选参数填写
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//modelview_matrix：用于构建观察相机及其视点的初始位置，http://imgtec.eetrend.com/blog/2019/100017250.html
        //相机、视点初始坐标 modelview_matrix对象
        // x、y、z：相机的初始坐标
        // lx、ly、lz：相机视点的初始位置，也即相机光轴朝向
        // up：相机自身那一轴朝上放置，可选如下参数填写：
        // pangolin::AxisX：X轴正方向
        // pangolin::AxisY：Y轴正方向
        // pangolin::AxisZ：Z轴正方向
        // pangolin::AxisNegX：X轴负方向
        // pangolin::AxisNegY：Y轴负方向
        // pangolin::AxisNegZ：Z轴负方向
//前三个参数依次为相机所在的位置，第四到第六个参数相机所看的视点位置(一般会设置在原点)，你可以用自己的脑袋当做例子，前三个参数告诉你脑袋在哪里，然后再告诉你看的东西在哪里，最后告诉你的头顶朝着哪里。
    );
    //在窗口中建立交互视图，用于显示相机观察到的信息内容
    pangolin::View & d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)//用于确定视图属性，其内参数如下：bottom、top：视图在视窗内的上下范围，依次为下、上，采用相对坐标表示（0：最下侧，1：最上侧），left、right：视图在视窗内的左右范围，依次为左、右，采用相对左边表示（0：最左侧，1：最右侧），aspect：视图的分辨率，也即分辨率 
    .SetHandler(new pangolin::Handler3D(s_cam));//用于确定视图的相机句柄

    while (pangolin::ShouldQuit() == false)//循环条件：视窗未被关闭
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//用于清空色彩缓冲区和深度缓冲区，刷新显示信息。若不使用清理，视窗将自动保留上一帧信息。
        d_cam.Activate(s_cam);//对交互视图对象进行激活相机
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//设置背景颜色，默认为黑色：    red、green、blue：颜色RGB数值，范围0~1,alpha：透明度
        //实现绘制线段：
        glLineWidth(2);//  设置大小
        for (size_t i = 0; i < poses.size(); i++)//size_t 类型表示C中任何对象所能达到的最大长度，它是无符号整数。它是为了方便系统之间的移植而定义的
        {
            // 画每个位姿的三个坐标轴
              //分别用红、绿、黄画出三个坐标轴
            Vector3d Ow = poses[i].translation();//返回当前变换平移部分的向量表示(可修改)，可以索引[]获取各分量,https://blog.csdn.net/sinat_38068956/article/details/119384933
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1,0,0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0,1,0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0,0,1));

            glBegin(GL_LINES);//  开始
            glColor3f(1.0,0.0,0.0);//  设置颜色
            glVertex3d(Ow[0],Ow[1],Ow[2]);//  设置起点、终点坐标
            glVertex3d(Xw[0],Xw[1],Xw[2]);
            glColor3f(0.0,1.0,0.0);//  设置颜色
            glVertex3d(Ow[0],Ow[1],Ow[2]);//  设置起点、终点坐标
            glVertex3d(Yw[0],Yw[1],Yw[2]);
            glColor3f(0.0,0.0,1.0);//  设置颜色
            glVertex3d(Ow[0],Ow[1],Ow[2]);//  设置起点、终点坐标
            glVertex3d(Zw[0],Zw[1],Zw[2]);
            glEnd();//  结束
        }

        //画出连线
        for (size_t i = 0; i < poses.size(); i++)
        {
            glColor3f(0.0,0.0,0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i] , p2 = poses[i+1];
            glVertex3d(p1.translation()[0],p1.translation()[1],p1.translation()[2]);
            glVertex3d(p2.translation()[0],p2.translation()[1],p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame(); //  帧循环，推进信息更迭
        usleep(5000);// sleep 5 ms,#include <unistd.h>,https://blog.csdn.net/napoleonwxu/article/details/46424309
        
    }
    
}
