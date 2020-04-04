#pragma once

#include <stdlib.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::string;
using namespace cv;
using namespace std;

void Evaluate(Mat standard, Mat myMap);