#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam {

/**
 * 配置类，使用SetParameterFile确定配置文件（读yaml文件）
 * 然后用Get得到对应值
 * 单例模式：https://www.runoob.com/design-pattern/singleton-pattern.html  ，  https://www.cnblogs.com/xyhj/p/14806383.html  ， https://zhuanlan.zhihu.com/p/37469260
 */
class Config {
   private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;//https://blog.csdn.net/qq_34122731/article/details/90722950

    Config() {}  // private constructor makes a singleton
   public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T Get(const std::string &key) {
        return T(Config::config_->file_[key]);
    }
};
}  // namespace myslam

#endif  // MYSLAM_CONFIG_H
