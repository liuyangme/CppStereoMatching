#include "Evaluate.h"

int main() {
	Mat groundtruth, mydepth;
	Mat errormap;

	groundtruth = imread("image/groundtruth/dispnoc0/000045_10.png", 0);
	mydepth = imread("image/mydepth/000045_10_64res_post-5.png", 0);

	if (groundtruth.empty() || mydepth.empty()) {
		cout << "image read error!" << endl;
		return -1;
	}

	errormap = Mat(groundtruth.rows, groundtruth.cols, CV_8UC3);

	for (int i = 0; i < groundtruth.rows; i++) {
		for (int j = 0; j < groundtruth.cols; j++) {
			groundtruth.ptr<uchar>(i)[j] = groundtruth.ptr<uchar>(i)[j] * 3;  //256--->64
		}
	}


	Evaluate(groundtruth, mydepth, errormap);

	return 0;
}




void Evaluate(Mat standard, Mat myMap, Mat errorMap) {
	//double sum = standard.rows * standard.cols;  //groundtruth图非稠密不可用此
	double sum = 0;
	int count = 0;
	int ans = 0;
	for (int i = 0; i < standard.rows; i++) {
		for (int j = 0; j < standard.cols; j++) {
			if (standard.ptr<uchar>(i)[j] != 0) {
				ans = standard.ptr<uchar>(i)[j] - myMap.ptr<uchar>(i)[j];
				sum++;  //计数扫描像素
				//与原图灰度相差大于1可认为是bad pixels，因为增强对比度，所以disparity maps都乘以3显示
				if (ans > 3 || ans < -3) {
					count++;
					//errorMap.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
					errorMap.ptr<Vec3b>(i)[j] = Vec3b(0, 0, 255);
				}
				else {
					//errorMap.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
					errorMap.ptr<Vec3b>(i)[j] = Vec3b(0, 255, 0);
				}
			}
		}
	}
	double result = (count + 0.0) / sum;
	cout << "Error rate = " << result * 100 << '%' << endl;
}