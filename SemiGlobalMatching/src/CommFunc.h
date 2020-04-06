#ifndef __COMMFUNC_H__
#define __COMMFUNC_H__

#include<opencv2/opencv.hpp>
using namespace cv;

//
// Opencv Lib 2.4.9
//
#ifdef _DEBUG
#pragma comment( lib, "opencv_calib3d249d.lib" )
#pragma comment( lib, "opencv_contrib249d.lib" )
#pragma comment( lib, "opencv_core249d.lib" )
#pragma comment( lib, "opencv_features2d249d.lib" )
#pragma comment( lib, "opencv_flann249d.lib" )
#pragma comment( lib, "opencv_gpu249d.lib" )
#pragma comment( lib, "opencv_highgui249d.lib" )
#pragma comment( lib, "opencv_imgproc249d.lib" )
#pragma comment( lib, "opencv_legacy249d.lib" )
#pragma comment( lib, "opencv_ml249d.lib" )
#pragma comment( lib, "opencv_nonfree249d.lib" )
#pragma comment( lib, "opencv_objdetect249d.lib" )
#pragma comment( lib, "opencv_photo249d.lib" )
#pragma comment( lib, "opencv_stitching249d.lib" )
#pragma comment( lib, "opencv_superres249d.lib" )
#pragma comment( lib, "opencv_ts249d.lib" )
#pragma comment( lib, "opencv_video249d.lib" )
#pragma comment( lib, "opencv_videostab249d.lib" )
#else
#pragma comment( lib, "opencv_calib3d249.lib" )
#pragma comment( lib, "opencv_contrib249.lib" )
#pragma comment( lib, "opencv_core249.lib" )
#pragma comment( lib, "opencv_features2d249.lib" )
#pragma comment( lib, "opencv_flann249.lib" )
#pragma comment( lib, "opencv_gpu249.lib" )
#pragma comment( lib, "opencv_highgui249.lib" )
#pragma comment( lib, "opencv_imgproc249.lib" )
#pragma comment( lib, "opencv_legacy249.lib" )
#pragma comment( lib, "opencv_ml249.lib" )
#pragma comment( lib, "opencv_nonfree249.lib" )
#pragma comment( lib, "opencv_objdetect249.lib" )
#pragma comment( lib, "opencv_photo249.lib" )
#pragma comment( lib, "opencv_stitching249.lib" )
#pragma comment( lib, "opencv_superres249.lib" )
#pragma comment( lib, "opencv_ts249.lib" )
#pragma comment( lib, "opencv_video249.lib" )
#pragma comment( lib, "opencv_videostab249.lib" )
#endif

// output matrix
template<class T>
void PrintMat(const Mat& mat)
{
	int rows = mat.rows;
	int cols = mat.cols;
	printf("\n%d x %d Matrix\n", rows, cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			cout << mat.at<T>(r, c) << "\t";
		}
		printf("\n");
	}
	printf("\n");
}

#endif
