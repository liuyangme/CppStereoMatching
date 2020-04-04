#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int isInside(Mat img, int i, int j);
Mat_<uchar> medianFilter(Mat_<uchar> img, int size);


