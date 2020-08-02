//
// Created by yuanhao on 20-6-12.
//

#ifndef LIVE_H
#define LIVE_H
#include <opencv2/core/mat.hpp>
#include "net.h"

struct ModelConfig {
    float scale;
    float shift_x;
    float shift_y;
    int height;
    int width;
    std::string name;
    bool org_resize;
};

struct LiveFaceBox {
    float x1;
    float y1;
    float x2;
    float y2;
};

class Live {
public:
    Live();

    ~Live();

    void LoadModel(std::vector<ModelConfig> &configs);

     float Detect(cv::Mat &src, LiveFaceBox &box);

private:
    cv::Rect CalculateBox(LiveFaceBox &box, int w, int h, ModelConfig &config);

private:
    std::vector<ncnn::Net *> nets_;
    std::vector<ModelConfig> configs_;
    const std::string net_input_name_ = "data";
    const std::string net_output_name_ = "softmax";
    int model_num_;
    int thread_num_;

    ncnn::Option option_;
};

#endif //LIVE_H
