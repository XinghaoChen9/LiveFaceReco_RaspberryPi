#include "arcface.h"
#include "livefacereco.hpp"

ncnn::Mat bgr2rgb(ncnn::Mat src)
{
    int src_w = src.w;
    int src_h = src.h;
    unsigned char* u_rgb = new unsigned char[src_w * src_h * 3];
    src.to_pixels(u_rgb, ncnn::Mat::PIXEL_BGR2RGB);
    ncnn::Mat dst = ncnn::Mat::from_pixels(u_rgb, ncnn::Mat::PIXEL_RGB, src_w, src_h);
    delete[] u_rgb;
    return dst;
}

Arcface::Arcface(string model_folder)
{
    string param_file = project_path+"/models/mobilefacenet/mobilefacenet.param";
    string bin_file = project_path+"/models/mobilefacenet/mobilefacenet.bin";

    this->net.load_param(param_file.c_str());
    this->net.load_model(bin_file.c_str());
}

Arcface::~Arcface()
{
    this->net.clear();
}

cv::Mat Arcface::getFeature(cv::Mat img)
{
    vector<float> feature;
    //cv to NCNN
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    //in = bgr2rgb(in);
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(this->feature_dim);
    for (int i = 0; i < this->feature_dim; i++)
        feature[i] = out[i];
    //normalize(feature);
    cv::Mat feature__=cv::Mat(feature,true);
    return feature__;
}

void Arcface::normalize(vector<float> &feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}


float calcSimilar(std::vector<float> feature1, std::vector<float> feature2)
{
    //assert(feature1.size() == feature2.size());
    float sim = 0.0;
    for (int i = 0; i < feature1.size(); i++)
        sim += feature1[i] * feature2[i];
    return sim;
}
