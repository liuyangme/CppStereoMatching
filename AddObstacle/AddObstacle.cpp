#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat srcImage = imread("srcImage/000117_10.png", 1);
    if (srcImage.empty() == 1) {
        printf("Error SrcImage! \t\n");
        return -1;
    }

	Mat dstImage = srcImage.clone();

	int rows_start = 100;
	int rows_end = 120;
	int cols_start = 100;
	int cols_end = 120;

	if ((rows_start > rows_end) || (cols_start > cols_end) || 
		(cols_end > dstImage.cols) || (rows_end > dstImage.rows)) {
		printf("Error Rows and Cols! \t\n");
		return -1;
	}

	for (int i= rows_start;i< rows_end;i++)
		for (int j = cols_start; j < cols_end; j++)
		{
			//图像通道判定
			if (dstImage.channels() == 1)
			{
				dstImage.at<uchar>(i, j) = 0;
			}
			else
			{
				dstImage.at<Vec3b>(i, j)[0] = 0;
				dstImage.at<Vec3b>(i, j)[1] = 0;
				dstImage.at<Vec3b>(i, j)[2] = 0;
			}
		}

    return 0;
}
