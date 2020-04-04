#include "medianFilter.h"

Mat_<uchar> medianFilter(Mat_<uchar> img, int size) {
	Mat_<uchar> res(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			std::vector<int> v;
			for (int u = -size / 2; u <= size / 2; u++) {
				for (int q = -size / 2; q <= size / 2; q++) {
					if (isInside(img, i + u, j + q)) {
						v.push_back(img(i + u, j + q));
					}
				}
			}
			std::sort(v.begin(), v.end());
			res(i, j) = v.at(v.size() / 2);
		}
	}
	return res;
}

int isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) {
		return 1;
	}
	return 0;
}