#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include "LidarIris.h"

using namespace std;

void OneCoupleCompare(string cloudFileName1, string cloudFileName2)
{
    LidarIris iris(4, 18, 1.6, 0.75, 50);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>), cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(cloudFileName1, *cloud0) == -1)
    {
        abort();
    }
    if (pcl::io::loadPCDFile(cloudFileName2, *cloud1) == -1)
    {
        abort();
    }
    clock_t startTime = clock();

    cv::Mat1b li1 = LidarIris::GetIris(*cloud0);
    cv::Mat1b li2 = LidarIris::GetIris(*cloud1);

    LidarIris::FeatureDesc fd1 = iris.GetFeature(li1);
    LidarIris::FeatureDesc fd2 = iris.GetFeature(li2);

    int bias;
    auto dis = iris.Compare(fd1, fd2, &bias);

    clock_t endTime = clock();

    cout << "try compare:" << endl
         << cloudFileName1 << endl
         << cloudFileName2 << endl;
    cout << "dis = " << dis << ", bias = " << bias << endl;
    cout << "times = " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "s."<< endl;

    cv::Mat1b img_iris, img_T;
    cv::vconcat(fd1.img, fd2.img, img_iris);
    cv::imshow("LiDAR Iris before transformation", img_iris);
    // cv::imwrite("../img/before.bmp", img_iris);
    
    cv::Mat temp = LidarIris::circShift(fd1.img, 0, bias);
    cv::vconcat(temp, fd2.img, img_iris);
    cv::imshow("LiDAR Iris after transformation", img_iris);
    // cv::imwrite("../img/after.bmp", img_iris);

    cv::hconcat(fd1.T, fd2.T, img_T);
    cv::imshow("LiDAR Iris Template", img_T);
    // cv::imwrite("../img/temp.bmp", img_T);

    cv::waitKey(0);
}

int main(int argc, char *argv[])
{

    if(argc != 3)
    {
        cerr << "usage: ./demo cloud1.pcd cloud2.pcd" << std::endl;
        return -1;
    }

    OneCoupleCompare(argv[1], argv[2]);
    return 0;
}