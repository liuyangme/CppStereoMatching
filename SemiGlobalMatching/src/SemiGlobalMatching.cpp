#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include"SGMfunction.h"

using namespace cv;
using namespace std;

int main() {
	Mat lImg = imread("image/leftimage/000146_10.png", 1);
	Mat rImg = imread("image/rightimage/000146_10.png", 1);
	if (!lImg.data || !rImg.data) {
		printf("Error: can not open image\n");
		printf("\nPress any key to continue...\n");
		system("pause");
		return -1;
	}
	int hei = lImg.rows;
	int wid = rImg.cols;
	Mat lDis = Mat::zeros(hei, wid, CV_32FC1);
	SGMStereo sgm;
	sgm.compute(lImg, rImg, lDis);
	imwrite("image/resimage/000146_10_64res.png", lDis);

	return 0;

}