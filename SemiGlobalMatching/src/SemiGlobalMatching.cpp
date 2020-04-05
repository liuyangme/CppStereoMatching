#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat lImg = imread("view1.png", 1);
	Mat rImg = imread("view5.png", 1);
	if (!lImg.data || !rImg.data) {
		printf("Error: can not open image\n");
		return -1;
	}

	return 0;
}


