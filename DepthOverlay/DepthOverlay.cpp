#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat SrcImage = imread("srcImage/000117_10.png", 1);
    Mat DepthImage = imread("srcImage/000117_10_res.png", 1);

    if (SrcImage.empty() == 1 || DepthImage.empty() == 1) {
        printf("Image Input Error!!");
        return -1;
    }

    Mat ColorRes, OverLapping;

    //伪彩色化
    applyColorMap(DepthImage, ColorRes, COLORMAP_HSV);
    //叠加
    addWeighted(ColorRes, 0.4, SrcImage, 1, 0, OverLapping);

    imwrite("resImage/000117_10.png", OverLapping);

    return 0;
}
