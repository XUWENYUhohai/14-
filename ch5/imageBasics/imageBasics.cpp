#include <iostream>
#include <chrono>//与时间日期有关的库

using namespace std;

#include <opencv2/core/core.hpp>//主要包含了opencv基本数据结构，动态数据结构，绘图函数，数组操作相关函数，辅助功能与系统函数和宏
#include <opencv2/highgui/highgui.hpp>//高层GUI图像交互模块 ： 主要包换了图形交互界面，媒体I/O的输入输出，视频信息的捕捉和提取，图像视频编码等。 

int main(int argc, char *argv[])
{
  // 读取argv[1]指定的图像
  cv::Mat image;//cv::Mat是OpenCV定义的用于表示任意维度的稠密数组，OpenCV使用它来存储和传递图像
  image = cv::imread(argv[1]);//cv::imread函数读取指定路径下的图像

  // 判断图像文件是否正确读取
  if(image.data == nullptr)
  {
    //数据不存在,可能是文件不存在
    cerr << "文件" << argv[1] << "不存在" << endl;
    return 0;
  }

  // 文件顺利读取, 首先输出一些基本信息
  cout << "图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << endl;
  cv::imshow("image",image);// 用cv::imshow显示图像
  cv::waitKey(0);// 暂停程序,等待一个按键输入
//无限延时
  // 判断image的类型
  //   CV_<bit_depth>(S|U|F)C<number_of_channels>

  // 1–bit_depth—比特数—代表8bite,16bites,32bites,64bites—举个例子吧–比如说,如
  //   如果你现在创建了一个存储–灰度图片的Mat对象,这个图像的大小为宽100,高100,那么,现在这张灰度图片中有10000个像素点，它每一个像素点在内存空间所占的空间大小是8bite,8位–所以它对应的就是CV_8

  // 2–S|U|F–S--代表—signed int—有符号整形
  //  U–代表–unsigned int–无符号整形
  //  F–代表–float---------单精度浮点型

  // 3–C<number_of_channels>----代表—一张图片的通道数,比如:
  //  1–灰度图片–grayImg—是–单通道图像
  //  3–RGB彩色图像---------是–3通道图像
  //  4–带Alph通道的RGB图像–是--4通道图像
  if (image.type() != CV_8UC1 && image.type() != CV_8UC3)
  {
    // 图像类型不符合要求
    cout << "请输入一张彩色图或灰度图." << endl;
    return 0;
  }

  // 遍历图像, 请注意以下遍历方式亦可使用于随机像素访问
  // 使用 std::chrono 来给算法计时
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (size_t y = 0; y < image.rows; y++)
  {
    // 用cv::Mat::ptr获得图像的行指针
    unsigned char * row_ptr = image.ptr<unsigned char>(y); // row_ptr是第y行的头指针
    for (size_t x = 0; x < image.cols; x++)
    {
      // 访问位于 x,y 处的像素
      unsigned char * data_ptr = &row_ptr[x * image.channels()];// data_ptr 指向待访问的像素数据
      // 输出该像素的每个通道,如果是灰度图就只有一个通道
      for (size_t c = 0; c != image.channels(); c++)
      {
        unsigned char data = data_ptr[c];// data为I(x,y)第c个通道的值
        //测试：unsigned char data = image.data[1];
        // 测试：unsigned char data = image.ptr(y)[x];
      } 
    }
  }
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout << "遍历图像用时：" << time_used.count() << "秒。" << endl;

  // 关于 cv::Mat 的拷贝
  // 直接赋值并不会拷贝数据
  cv::Mat image_another = image;
  // 修改 image_another 会导致 image 发生变化
  ////参数1和参数2：左上角点坐标；参数3和参数4：宽和高,https://blog.51cto.com/u_11531789/4970233
  image_another(cv::Rect(0,0,100,100)).setTo(0);// 将左上角100*100的块置零
  cv::imshow("image",image);
  cv::waitKey(0);

  // 使用clone函数来拷贝数据
  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0,0,100,100)).setTo(255);
  cv::imshow("image",image);
  cv::imshow("image_clone",image_clone);
  cv::waitKey(0);//== cv::waitKey();

  // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
  cv::destroyAllWindows();
  return 0;
}