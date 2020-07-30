#pragma once

#include <stdlib.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void KITTIEvaluate(Mat standard, Mat myMap, Mat errorMap);