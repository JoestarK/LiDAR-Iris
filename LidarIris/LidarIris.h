#ifndef _LIDAR_IRIS_H_
#define _LIDAR_IRIS_H_

#include <vector>
#include <flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class LidarIris
{
public:
    struct FeatureDesc
    {
        cv::Mat1b img;
        cv::Mat1b T;
        cv::Mat1b M;
    };

    LidarIris(int nscale, int minWaveLength, float mult, float sigmaOnf, int matchNum) : _nscale(nscale),
                                                                                         _minWaveLength(minWaveLength),
                                                                                         _mult(mult),
                                                                                         _sigmaOnf(sigmaOnf),
                                                                                         _matchNum(matchNum),
                                                                                         vecList(flann::Index<flann::L2<float>>(flann::KDTreeIndexParams(4))),
                                                                                         // indicesBuffer(std::vector<int>(matchNum)),
                                                                                         // distsBuffer(std::vector<float>(matchNum)),
                                                                                         indices(flann::Matrix<int>(new int[matchNum], 1, matchNum)),
                                                                                         dists(flann::Matrix<float>(new float[matchNum], 1, matchNum))
    {
    }
    LidarIris(const LidarIris &) = delete;
    LidarIris &operator=(const LidarIris &) = delete;

    static cv::Mat1b GetIris(const pcl::PointCloud<pcl::PointXYZ> &cloud);
    //
    void UpdateFrame(const cv::Mat1b &frame, int frameIndex, float *matchDistance, int *matchIndex);
    //
    float Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias = nullptr);
    //
    FeatureDesc GetFeature(const cv::Mat1b &src);
    FeatureDesc GetFeature(const cv::Mat1b &src, std::vector<float> &vec);
    std::vector<cv::Mat2f> LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf);
    void GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias);
    //
    static inline cv::Mat circRowShift(const cv::Mat &src, int shift_m_rows);
    static inline cv::Mat circColShift(const cv::Mat &src, int shift_n_cols);
    static cv::Mat circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols);

private:
    void LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M);

    int _nscale;
    int _minWaveLength;
    float _mult;
    float _sigmaOnf;
    int _matchNum;

    flann::Index<flann::L2<float>> vecList;
    std::vector<FeatureDesc> featureList;
    std::vector<int> frameIndexList;
    flann::Matrix<int> indices;
    flann::Matrix<float> dists;
    // std::vector<int> indicesBuffer;
    // std::vector<float> distsBuffer;
};

#endif
