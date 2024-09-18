#include <iostream>
#include <iomanip>//主要是对cin,cout之类的一些操纵运算子，比如setfill,setw,setbase,setprecision等等。它是I/O流控制头文件,就像C里面的格式化输出一样

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <pangolin/pangolin.h>

struct RotationMatrix
{
    Matrix3d matrix = Matrix3d::Identity();
};

ostream &operator<<(ostream & out,const RotationMatrix &r)
{
    out.setf(ios::fixed);
    Matrix3d matrix = r.matrix;
  out << '=';
  out << "[" << setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
      << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
      << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";
  return out;
}

istream &operator>>(istream &in, RotationMatrix &r) {
  return in;
}

struct TranslationVector {
  Vector3d trans = Vector3d(0, 0, 0);
};

ostream &operator<<(ostream &out, const TranslationVector &t) {
  out << "=[" << t.trans(0) << ',' << t.trans(1) << ',' << t.trans(2) << "]";
  return out;
}

istream &operator>>(istream &in, TranslationVector &t) {
  return in;
}

struct QuaternionDraw {
  Quaterniond q;
};

ostream &operator<<(ostream &out, const QuaternionDraw quat) {
  auto c = quat.q.coeffs();
  out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
  return out;
}

istream &operator>>(istream &in, const QuaternionDraw quat) {
  return in;
}

int main(int argc, char *argv[])//https://blog.csdn.net/weixin_45929038/article/details/122911049
{
      pangolin::CreateWindowAndBind("visualize geometry", 1000, 600);//该函数用于创建一个指定大小、名称的GUI窗口。
  glEnable(GL_DEPTH_TEST);//开启深度测试:深度测试可以使得Pangolin构建的相机只展示镜头朝向一侧的像素信息，从而避免容易混淆的透视关系。
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
    pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)//http://imgtec.eetrend.com/blog/2019/100017250.html
    //前三个参数依次为相机所在的位置，第四到第六个参数相机所看的视点位置(一般会设置在原点)，你可以用自己的脑袋当做例子，前三个参数告诉你脑袋在哪里，然后再告诉你看的东西在哪里，最后告诉你的头顶朝着哪里。
  );

  const int UI_WIDTH = 500;//  交互视图同控制面板分界

  pangolin::View &d_cam = pangolin::CreateDisplay().//  右侧:交互视图
    SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f).//定义一个常量UI_WITH作为两者的分界面，随后使用pangolin::Attach::Pix()获得对应的像素信息。
    SetHandler(new pangolin::Handler3D(s_cam));

  // ui在pcl中建立 R T euler q 列表，Pangolin将所有“控件”视为一个对象pangolin::Var，可用于构建按钮、滑条、控制对象、信息显示等内容：
  pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
  pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
  pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
  pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));// 左侧：控制面板

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();//模型可视化矩阵赋值给matrix
    Matrix<double, 4, 4> m = matrix;

    RotationMatrix R;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R.matrix(i, j) = m(j, i);//取出m中的旋转矩阵
    rotation_matrix = R;

    TranslationVector t;
    t.trans = Vector3d(m(0, 3), m(1, 3), m(2, 3));//取出m的平移向量
    t.trans = -R.matrix * t.trans;
    translation_vector = t;

    TranslationVector euler;
    euler.trans = R.matrix.eulerAngles(2, 1, 0);
    euler_angles = euler;//得到欧拉角

    QuaternionDraw quat;
    quat.q = Quaterniond(R.matrix);//得到四元数
    quaternion = quat;

    glColor3f(1.0, 1.0, 1.0);

    pangolin::glDrawColouredCube();//在原点处画立方体
    // draw the original axis // 画原始坐标轴
    glLineWidth(3);//绘制线段//  设置大小
    glColor3f(0.8f, 0.f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(10, 0, 0);
    glColor3f(0.f, 0.8f, 0.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 10, 0);
    glColor3f(0.2f, 0.2f, 1.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 10);
    glEnd();

    pangolin::FinishFrame();
  }
    return 0;
}

