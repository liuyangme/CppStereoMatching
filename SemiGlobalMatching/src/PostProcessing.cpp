#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "PostProcessing.h"

using namespace cv;
using namespace std;


void PostMedianFlitering(const Mat& src, Mat& dst) {
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<float>(i, j) = PostMedian(src.at<float>(i, j), src.at<float>(i + 1, j + 1),
					src.at<float>(i + 1, j), src.at<float>(i, j + 1), src.at<float>(i + 1, j - 1),
					src.at<float>(i - 1, j + 1), src.at<float>(i - 1, j), src.at<float>(i, j - 1),
					src.at<float>(i - 1, j - 1));
			}
			else
				_dst.at<float>(i, j) = src.at<float>(i, j);
		}
	_dst.copyTo(dst);//øΩ±¥
}

float PostMedian(float n1, float n2, float n3, float n4, float n5,
	float n6, float n7, float n8, float n9) {
	float arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//œ£∂˚≈≈–Ú
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//∑µªÿ÷–÷µ
}

