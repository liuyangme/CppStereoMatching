#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

//主要是RGB图像和视差图像的数据类型有出入，后续考虑函数复用进行合并简化
void PreMedianFlitering(const Mat& src, Mat& dst);
uchar PreMedian(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9);

void PostMedianFlitering(const Mat& src, Mat& dst);
float PostMedian(float n1, float n2, float n3, float n4, float n5,
	float n6, float n7, float n8, float n9);