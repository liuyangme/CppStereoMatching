/*
This project implements the methods in the following paper. Please cite this paper, depending on the use.

Xing Mei, Xun Sun, Weiming Dong, Haitao Wang and Xiaopeng Zhang. Segment-Tree based Cost Aggregation for Stereo Matching, in CVPR 2013.

The code is written by Yan Kong, <kongyanwork@gmail.com>, 2013.

LICENSE
Copyright (C) 2012-2013 by Yan Kong
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "StereoDisparity.h"

using namespace std;
using namespace cv;

//std::string getFileName(const char* path) {
//	std::string str(path);
//	int pos = str.rfind('\\');
//	return str.substr(pos + 1, str.length() - pos);
//}

int main(int argc, const char** argv) {
	
	//if (argc < 4) {
	//	const char* head = "*****Segment-Tree based Cost Aggregation for Stereo Matching[CVPR2013]*****\n\n";
	//	printf("%s", head);
	//	printf("Usage:\n%s leftImgPath rightImgPath dispImgPath [maxLevel] [scale] [sigma] [method]\n", getFileName(argv[0]).c_str());
	//	printf("maxDispLevel: default 60\nscale: default 4\nsigma: default 0.1\nnmethod: 0(defalut, ST-1) or 1(ST-2)\n");
	//	exit(0);
	//}

	int maxLevel = 128;
	int scale = 4;
	float sigma = 0.1f;
	//METHOD method = ST_RAW;
	METHOD method = ST_REFINED;



	argv[1] = "image/leftimage/000146_11.png";
	argv[2] = "image/rightimage/000146_11.png";
	argv[3] = "image/000146_11_result.png";

	stereo_routine(argv[1], argv[2], argv[3], maxLevel, scale, sigma, method);

	return 0;
}
