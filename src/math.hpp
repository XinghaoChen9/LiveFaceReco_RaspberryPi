#ifndef __MATH__HEADER__
#define __MATH__HEADER__

#include <opencv4/opencv2/opencv.hpp>

namespace Statistics
{
    #define PI 3.14159265
    using namespace cv;
    /**
     * This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
     * original distribution in order to make two feature more distinguishable.
     * mean value is set to 0 and std is set to 1
    */
    cv::Mat zScore(const cv::Mat &fc) 
    {
        cv::Mat mean, std;
        cv::meanStdDev(fc, mean, std);
        //cout <<"mean is :"<< mean <<"std is :"<< std << endl;
        cv::Mat fc_norm = (fc - mean) / std;
        return fc_norm;
    }

    /**
    * This module is using to computing the cosine distance between input feature and ground truth feature
    *  */
    inline float cosineDistance(const cv::Mat &v1, const cv::Mat &v2) {
        double dot = v1.dot(v2);
        double denom_v1 = norm(v1);
        double denom_v2 = norm(v2);
        return dot / (denom_v1 * denom_v2);
    }

    /**
     * Calculating the turning angle of face
     *  */
    inline double countAngle(float landmark[5][2]) {
        double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
        double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
        double angle = atan(abs(b) / a) * 180 / PI;
        return angle;
    }
}
#endif