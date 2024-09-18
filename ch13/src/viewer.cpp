//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {

Viewer::Viewer() {
    //std::function:     https://dabaojian.blog.csdn.net/article/details/49134235?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-49134235-blog-115210731.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-49134235-blog-115210731.pc_relevant_aa&utm_relevant_index=2
    //std::bind:         https://www.zhihu.com/question/68128328/answer/2373852519
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));//类的成员函数必须通过类的对象或者指针调用，因此在bind时，bind的第一个参数的位置来指定一个类的实列、指针或引用
}

void Viewer::Close() {
    viewer_running_ = false;
    viewer_thread_.join();//不能马上毙掉该线程，因为pangolin显示还在运行，当我们设置为标志为false时，pangolin的while循环会结束同时执行退出程序，打印STOP VIEWER,所以这一步是线程.join函数获得cpu优先执行权，其他线程停止等待该线程运行结束--即线程循环中的pangolin那边的运行结束
}

void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);//避免前端又加入了新的一帧
    current_frame_ = current_frame;
}

void Viewer::UpdateMap() {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    assert(map_ != nullptr);
    active_keyframes_ = map_->GetActiveKeyFrames();
    //active_landmarks_ = map_->GetActiveMapPoints();
     active_landmarks_ = map_->GetAllMapPoints();// 这样可以显示所有地图点，同时也能够看出没有回环检测，累计误差很大
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    pangolin::CreateWindowAndBind("MySLAM", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(//构建观察相机对象
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));//相机自身y轴朝下（世界坐标y轴的反方向）,http://imgtec.eetrend.com/blog/2019/100017250.html
        //前三个参数依次为相机所在的位置，第四到第六个参数相机所看的视点位置(一般会设置在原点)，你可以用自己的脑袋当做例子，前三个参数告诉你脑袋在哪里，然后再告诉你看的东西在哪里，最后告诉你的头顶朝着哪里。

    // Add named OpenGL viewport to window and provide 3D Handler//在窗口中建立交互视图，用于显示相机观察到的信息内容
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    while (!pangolin::ShouldQuit() && viewer_running_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        if (current_frame_) {
            DrawFrame(current_frame_, green);
            FollowCurrentFrame(vis_camera);

            cv::Mat img = PlotFrameImage();
            cv::imshow("image", img);
            cv::waitKey(1);
        }

        if (map_) {
            DrawMapPoints();
        }

        pangolin::FinishFrame();
        usleep(5000);
    }

    LOG(INFO) << "Stop viewer";
}

cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out;
    cv::cvtColor(current_frame_->left_img_, img_out, CV_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.lock()) {
            auto feat = current_frame_->features_left_[i];
            cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0),
                       2);
        }
    }
    return img_out;
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    SE3 Twc = current_frame_->Pose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);//其实现原理是Tc0_w *Tw_c1 = Tc0_c1  ，Tc0_w表示前一时刻的位姿，这样就能得到当前帧和上一帧的位姿变换矩阵，使用follow函数进行更新和跟踪（应该就是运行时视角不断变化的原因）。
}

void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    SE3 Twc = frame->Pose().inverse();
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();//https://www.ngui.cc/51cto/show-3379.html     ,当前矩阵 = 模型视图矩阵（http://t.zoukankan.com/mtcnn-p-9411922.html） ， 一开始这个应该就是指世界坐标原点吧，后来每次就是当前帧的相机坐标相对于世界坐标系

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();//Eigen::Matrix4f底层仍是Matrix<float,4,4>是模板类，所以https://www.codenong.com/30339239/  ， https://blog.csdn.net/keineahnung2345/article/details/119964609
    //.data():指向此矩阵的数据数组的指针
    glMultMatrixf((GLfloat*)m.data());//右乘当前矩阵：https://docs.microsoft.com/zh-cn/windows/win32/opengl/glmultmatrixf?redirectedfrom=MSDN     ,OpenGL中的模型视图变换矩阵全是(顶点对象)右乘当前变换矩阵

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);//这里的坐标应该是相对于当前帧坐标系的,原来ch3中的plotTrajectory.cpp是Two，现在是Tow右乘Twc
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();//对应 glPushMatrix()
}

void Viewer::DrawMapPoints() {
    const float red[3] = {1.0, 0, 0};
    for (auto& kf : active_keyframes_) {
        DrawFrame(kf.second, red);
    }

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto& landmark : active_landmarks_) {
        auto pos = landmark.second->Pos();
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

}  // namespace myslam
