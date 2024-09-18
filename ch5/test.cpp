#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    cv::Mat image = cv::imread("./ch5/rgbd/color/1.png");
    cv::Mat test(image.rows,image.cols,image.type());
    for (size_t v = 0; v < image.rows; v++)
    {
        for (size_t u = 0; u < image.cols; u++)
        {   //1.
            //test.at<cv::Vec3b>(v,u)[0] = image.at<cv::Vec3b>(v,u)[0];
            // test.at<cv::Vec3b>(v,u)[1] = image.at<cv::Vec3b>(v,u)[1];
            // test.at<cv::Vec3b>(v,u)[2] = image.at<cv::Vec3b>(v,u)[2];
            //2.
            //test.at<cv::Vec3b>(v,u) = image.at<cv::Vec3b>(v,u);
            //3.
            // test.data[v * test.step + u * test.channels()] = image.data[v * image.step + u * image.channels()];
            // test.data[v * test.step + u * test.channels()+1] = image.data[v * image.step + u * image.channels()+1];
            // test.data[v * test.step + u * test.channels()+2] = image.data[v * image.step + u * image.channels()+2];
            //4.
            //test.ptr(v)[u] = image.ptr(v)[u];不行，因为Mat里一行是cols * channels（）的，（u）[v]的一行只是0-cols的部分
            test.ptr(v)[u * image.channels()] = image.ptr(v)[u * image.channels()];
            test.ptr(v)[u * image.channels()+1] = image.ptr(v)[u * image.channels()+1];
            test.ptr(v)[u * image.channels()+2] = image.ptr(v)[u * image.channels()+2];
        }
        
    }
    cv::imshow("image",image);
    cv::imshow("test",test);
    cv::waitKey();
    return 0;
}
