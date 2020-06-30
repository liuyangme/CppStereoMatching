#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string>
#include <iostream>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PixelBasedCost.h"
#include "ConstBlockAggregation.h"
#include "WTA.h"
#include "RankFilter.h"
#include "CensusTransform.h"
#include "ConvolveImage.h"
#include "NCCDisparity.h"
#include "CrossCheck.h"
#include "AdaptiveWeightAggreagtion.h"
#include "CrossBasedAggregation.h"
#include "CombineCost.h"
#include "medianFilter.h"
#include "Evaluate.h"


using std::string;
using namespace cv;
using namespace std;

#define square(x) x*x
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5)) 
#define FileNumber 1

////////////////////////////////////////////////////////////////
/////////////////////FUNCTION DECLERATIONS//////////////////////
////////////////////////////////////////////////////////////////
void scalecvMat(unsigned char* src, int rdim, int cdim);
void scalecvMat(unsigned char* src, int rdim, int cdim, float coeff);

int main(int argc, char** argv)
{
	try {
		/////////////////////////////////////////////////////////////////////////
		///////////////Initialization////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////
		//Input(s)

		//文件夹下的目录
		//string name[FileNumber] = { "Aloe", "Baby1", "Baby2", "Baby3", "Bowling1",
		//"Bowling2", "Cloth1", "Cloth2", "Cloth3", "Cloth4", "Flowerpots",
		//"Lampshade1", "Lampshade2", "Midd1", "Midd2", "Monopoly",
		//"Plastic", "Rocks1", "Rocks2", "Wood1", "Wood2" };
	

		const time_t start = time(NULL);

		for (int i = 0; i < FileNumber; i++) {

			//读取images文件夹中的源图片
			//Mat left_image = imread("images/" + name[i] + "/view1.png", 0);
			//Mat right_image = imread("images/" + name[i] + "/view5.png", 0);

			Mat left_image = imread("images/leftimage/000045_11.png", 0);
			Mat right_image = imread("images/rightimage/000045_11.png", 0);

			if ((!left_image.data) || (!right_image.data)) {
				printf("Please input vaild image!\n");
				return -1;
			}

			unsigned char* ref_image = left_image.data;
			unsigned char* tar_image = right_image.data;

			//自适应窗口聚合彩色图像输入
			Mat RGB_left_image, RGB_right_image;
			Mat LAB_left_image, LAB_right_image;
			unsigned char* RGB_left_pnt, * RGB_right_pnt;
			unsigned char* LAB_left_pnt, * LAB_right_pnt;
			//RGB_left_image = imread("images/" + name[i] + "/view1.png", 1);
			//RGB_right_image = imread("images/" + name[i] + "/view5.png", 1);
			RGB_left_image = imread("images/leftimage/000045_11.png", 1);
			RGB_right_image = imread("images/rightimage/000045_11.png", 1);
			RGB_left_pnt = RGB_left_image.data;
			RGB_right_pnt = RGB_right_image.data;
			int rdim = left_image.rows;
			int cdim = left_image.cols;

			//匹配参数选择
			int dispRange = 64;//disparity range
			int costChoice1 = 3;//(1) AD, (2) Rank, (3) Census, (4) LoG 
			int costChoice2 = 0;//(0) Single Cost (1) AD, (2) Rank, (3) Census (4) LoG
			int aggrChoice = 3;//(1) Block, (2) Adaptive Weight, (3) Cross-Based
			int autoScaleOrNot = 1;//1:Yes
			float lambdaFirst = 1;
			float lambdaSecond = 1;
			//Aggregation Window Radius
			int mask_Rad = 1;
			//Census Mask Initialization
			int maskHrad = 5;
			int maskWrad = 3;
			int maskT = (2 * maskHrad + 1) * (2 * maskWrad + 1);
			//Rank Mask Initialization
			int maskRankSize = 7;
			//LoG Mask Initialization
			float F_LOG[25] = { 0.0239, 0.046, 0.0499, 0.046, 0.0239,
								0.046, 0.0061, -0.0923, 0.0061, 0.046,
								0.0499, -0.0923, -0.3182, -0.0923, 0.0499,
								0.046, 0.0061, -0.0923, 0.0061, 0.046,
								0.0239, 0.046, 0.0499, 0.046, 0.0239 };
			//Adaptive-Weight Parameters
			int adapWinRad = 16;
			float gamma_similarity = 7.0f;
			float gamma_proximity = 36.0f;
			int adapWinEdge = 2 * adapWinRad + 1;
			int adapWinArea = square(adapWinEdge);
			//Cross-Based Aggregation Parameters
			float trunc = 24;  //24?

			if (aggrChoice == 2) {
				cvtColor(RGB_left_image, LAB_left_image, COLOR_BGR2Lab);
				cvtColor(RGB_right_image, LAB_right_image, COLOR_BGR2Lab);
				LAB_left_pnt = LAB_left_image.data;
				LAB_right_pnt = LAB_right_image.data;
			}

			/////////////////////////////////////////////////////////////////////////
			////////////////////////Host Memory Allocation///////////////////////////
			/////////////////////////////////////////////////////////////////////////
			Mat show_image_L(rdim, cdim, CV_8UC1, Scalar(0));
			Mat show_image_R(rdim, cdim, CV_8UC1, Scalar(0));
			float* cost_image1_L, * cost_image1_R, * cost_image2, * cost_new_image_L, * cost_new_image_R, * cost_;
			unsigned char* rankFiltImageRef, * rankFiltImageTar;
			float* LoGFiltImageRef, * LoGFiltImageTar;
			bool* censusRefImage, * censusTarImage;
			float* weightsLeft, * weightsRight, * proxWeight;
			int* HorzOffset_ref, * HorzOffset_tar, * VertOffset_ref, * VertOffset_tar;
			/////////////////////////////////////////////////////
			unsigned char* shwImPntL = show_image_L.data;
			unsigned char* shwImPntR = show_image_R.data;
			unsigned char* ref_disparity = (unsigned char*)malloc(rdim * cdim * sizeof(unsigned char));
			//unsigned char * tar_disparity = (unsigned char *)malloc( rdim * cdim * sizeof(unsigned char));
			cost_image1_L = (float*)malloc(rdim * cdim * dispRange * sizeof(float));
			cost_image1_R = (float*)malloc(rdim * cdim * dispRange * sizeof(float));
			if (costChoice2 != 0)
				cost_image2 = (float*)malloc(rdim * cdim * dispRange * sizeof(float));
			if (aggrChoice == 3 || aggrChoice == 2) {
				cost_new_image_L = (float*)malloc(rdim * cdim * dispRange * sizeof(float));
				cost_new_image_R = (float*)malloc(rdim * cdim * dispRange * sizeof(float));
			}
			//unsigned char * cross_check_image = (unsigned char *)malloc(rdim * cdim * sizeof(unsigned char));
			if (costChoice1 == 2 || costChoice2 == 2) {
				rankFiltImageRef = (unsigned char*)malloc(rdim * cdim * sizeof(unsigned char));
				rankFiltImageTar = (unsigned char*)malloc(rdim * cdim * sizeof(unsigned char));
			}
			if (costChoice1 == 4 || costChoice2 == 4) {
				LoGFiltImageRef = (float*)malloc(rdim * cdim * sizeof(float));
				LoGFiltImageTar = (float*)malloc(rdim * cdim * sizeof(float));
			}
			if (costChoice1 == 3 || costChoice2 == 3) {
				censusRefImage = (bool*)malloc(rdim * cdim * maskT * sizeof(bool));
				censusTarImage = (bool*)malloc(rdim * cdim * maskT * sizeof(bool));
			}
			if (aggrChoice == 2) {
				proxWeight = (float*)malloc(adapWinArea * sizeof(float));
				weightsLeft = (float*)malloc(rdim * cdim * adapWinArea * sizeof(float));
				weightsRight = (float*)malloc(rdim * cdim * adapWinArea * sizeof(float));
				memset(weightsLeft, 0, rdim * cdim * adapWinArea * sizeof(float));
				memset(weightsRight, 0, rdim * cdim * adapWinArea * sizeof(float));
			}
			if (aggrChoice == 3) {
				HorzOffset_ref = (int*)malloc(rdim * cdim * 2 * sizeof(int));
				HorzOffset_tar = (int*)malloc(rdim * cdim * 2 * sizeof(int));
				VertOffset_ref = (int*)malloc(rdim * cdim * 2 * sizeof(int));
				VertOffset_tar = (int*)malloc(rdim * cdim * 2 * sizeof(int));
			}

			switch (costChoice1) {
			case 1:
				//AD
				//if (aggrChoice != 3)
				//	pixelBasedCostL2R_Color(RGB_left_pnt, RGB_right_pnt, cost_image1_L, dispRange, rdim, cdim, 0);
				//else
				//	pixelBasedCostL2R_Color(RGB_left_pnt, RGB_right_pnt, cost_image1_L, dispRange, rdim, cdim, trunc, 0);
				break;
			case 2:
				//Rank
				//calculateRankFilt(ref_image, rankFiltImageRef, maskRankSize, rdim, cdim);
				//calculateRankFilt(tar_image, rankFiltImageTar, maskRankSize, rdim, cdim);
				//pixelBasedL2R(rankFiltImageRef, rankFiltImageTar, cost_image1_L, dispRange, rdim, cdim);
				break;
			case 3:
				//Census
				calcCensusTrans(ref_image, censusRefImage, maskHrad, maskWrad, rdim, cdim);
				calcCensusTrans(tar_image, censusTarImage, maskHrad, maskWrad, rdim, cdim);
				calcHammingDist_L(censusRefImage, censusTarImage, cost_image1_L, maskHrad, maskWrad, rdim, cdim, dispRange);
				calcHammingDist_R(censusTarImage, censusRefImage, cost_image1_R, maskHrad, maskWrad, rdim, cdim, dispRange);
				break;
			case 4:
				//LoG
				//convolveImage(ref_image, LoGFiltImageRef, F_LOG, rdim, cdim, 2, 2);
				//convolveImage(tar_image, LoGFiltImageTar, F_LOG, rdim, cdim, 2, 2);
				//pixelBasedCostL2R_Float(LoGFiltImageRef, LoGFiltImageTar, cost_image1_L, dispRange, rdim, cdim);
				break;
			}

			if (costChoice2 != 0) {
				//switch (costChoice2) {
				//case 1:
				//	//AD
				//	pixelBasedCostL2R_Color(RGB_left_pnt, RGB_right_pnt, cost_image2, dispRange, rdim, cdim, 0);
				//	break;
				//case 2:
				//	//Rank
				//	calculateRankFilt(ref_image, rankFiltImageRef, maskRankSize, rdim, cdim);
				//	calculateRankFilt(tar_image, rankFiltImageTar, maskRankSize, rdim, cdim);
				//	pixelBasedL2R(rankFiltImageRef, rankFiltImageTar, cost_image2, dispRange, rdim, cdim);
				//	break;
				//case 3:
				//	//Census
				//	calcCensusTrans(ref_image, censusRefImage, maskHrad, maskWrad, rdim, cdim);
				//	calcCensusTrans(tar_image, censusTarImage, maskHrad, maskWrad, rdim, cdim);
				//	calcHammingDist_L(censusRefImage, censusTarImage, cost_image2, maskHrad, maskWrad, rdim, cdim, dispRange);
				//	break;
				//case 4:
				//	//LoG
				//	convolveImage(ref_image, LoGFiltImageRef, F_LOG, rdim, cdim, 2, 2);
				//	convolveImage(tar_image, LoGFiltImageTar, F_LOG, rdim, cdim, 2, 2);
				//	pixelBasedCostL2R_Float(LoGFiltImageRef, LoGFiltImageTar, cost_image2, dispRange, rdim, cdim);
				//	break;
				//}
				////Combine costs
				//combineCost(cost_image1_L, cost_image2, lambdaFirst, lambdaSecond, rdim, cdim, dispRange);  //cost_image1=A*cost_image1+B*cost_image2
			}

			switch (aggrChoice) {
			case 1:
				//constant box
				//boxAggregate(cost_image1_L, rdim, cdim, dispRange, mask_Rad);
				break;
			case 2:
				//Adaptive Weighting
				//calcProxWeight(proxWeight, gamma_proximity, adapWinRad);
				//computeWeights(weightsLeft, proxWeight, LAB_left_pnt, gamma_similarity, rdim, cdim, adapWinRad);
				//computeWeights(weightsRight, proxWeight, LAB_right_pnt, gamma_similarity, rdim, cdim, adapWinRad);
				//calcAWCostL2R(weightsLeft, weightsRight, cost_image1_L, cost_new_image, rdim, cdim, dispRange, adapWinRad);
				//cost_image1_L = cost_new_image;
				break;
			case 3:
				findCross(RGB_left_pnt, HorzOffset_ref, VertOffset_ref, rdim, cdim);
				findCross(RGB_right_pnt, HorzOffset_tar, VertOffset_tar, rdim, cdim);
				aggregate_cost_horizontal(cost_image1_L, rdim, cdim, dispRange);
				aggregate_cost_horizontal(cost_image1_R, rdim, cdim, dispRange);
				cross_stereo_aggregation(cost_image1_L, cost_new_image_L, HorzOffset_ref, HorzOffset_tar, VertOffset_ref, VertOffset_tar, rdim, cdim, dispRange);
				cross_stereo_aggregation(cost_image1_R, cost_new_image_R, HorzOffset_ref, HorzOffset_tar, VertOffset_ref, VertOffset_tar, rdim, cdim, dispRange);
				cost_image1_L = cost_new_image_L;
				cost_image1_R = cost_new_image_R;
				break;
			}

			WTA(cost_image1_L, shwImPntL, rdim, cdim, dispRange);
			WTA(cost_image1_R, shwImPntR, rdim, cdim, dispRange);
			if (autoScaleOrNot == 1) {
				scalecvMat(shwImPntL, rdim, cdim);
				scalecvMat(shwImPntR, rdim, cdim);
			}
			else {
				if (dispRange == 16) {
					scalecvMat(shwImPntL, rdim, cdim, 16);
				}
				else if (dispRange == 20) {
					scalecvMat(shwImPntL, rdim, cdim, 8);
				}
				else if (dispRange == 60) {
					scalecvMat(shwImPntL, rdim, cdim, 4);
				}
				else {
					printf("Invalid disparity range! \n");
					return 0;
				}
			}

			Mat show_image_color_L, show_image_color_R;
			//normalize(show_image_L, show_image_L, 0, 255, NORM_MINMAX);
			//normalize(show_image_R, show_image_R, 0, 255, NORM_MINMAX);
			show_image_L = medianFilter(show_image_L, 3);
			show_image_R = medianFilter(show_image_R, 3);
			applyColorMap(show_image_L, show_image_color_L, COLORMAP_RAINBOW);
			applyColorMap(show_image_R, show_image_color_R, COLORMAP_RAINBOW);
			//imwrite("resultImages/" + name[i] + "/" + name[i] + "_disp1_L.png", show_image_color_L);
			//imwrite("resultImages/" + name[i] + "/" + name[i] + "_disp5_R.png", show_image_color_R);
			//cout << "Matching " + name[i] + " is OK!\n\r";
			imwrite("resultImages/000045_11_disp_64CCL.png", show_image_L);
			imwrite("resultImages/000045_11_disp_64CCR.png", show_image_R);

			//视差评估
			//Mat standardLeft = imread("images/" + name[i] + "/disp1.png", -1);
			//cout << name[i] << "_Left_ASW: ";
			//Evaluate(standardLeft, show_image_L);
			//Mat standardRight = imread("images/" + name[i] + "/disp5.png", -1);
			//cout << name[i] << "_Right_ASW: ";
			//Evaluate(standardRight, show_image_R);

			//释放内存
			free(cost_image1_L);
			free(cost_image1_R);
			//free(cost_new_image_L);
			//free(cost_new_image_R);
			free(censusRefImage);
			free(censusTarImage);
			free(HorzOffset_ref);
			free(HorzOffset_tar);
			free(VertOffset_ref);
			free(VertOffset_tar);
		}
			waitKey(0);
	}
	catch (string msg) {
		printf("ERR:%s\n", msg.c_str());
		printf("Error catched\n");
	}
	return 1;
}

//////////////////////////////////////////////////////// FUNCTIONS//////////////////////////////////////////////

void scalecvMat(unsigned char* src, int rdim, int cdim) {
	unsigned char upprBound = 0;
	float coeff = 0, res = 0;
	for (int y = 0; y < rdim; y++)
		for (int x = 0; x < cdim; x++) {
			if (src[y * cdim + x] > upprBound)
				upprBound = src[y * cdim + x];
		}
	coeff = (float)255 / (float)upprBound;
	for (int y = 0; y < rdim; y++)
		for (int x = 0; x < cdim; x++) {
			res = round((float)(src[y * cdim + x]) * coeff);
			res = (res > 255) ? 255 : res;
			src[y * cdim + x] = (unsigned char)res;
		}
}

void scalecvMat(unsigned char* src, int rdim, int cdim, float coeff) {
	float res = 0;
	for (int y = 0; y < rdim; y++)
		for (int x = 0; x < cdim; x++) {
			res = round((float)(src[y * cdim + x]) * coeff);
			res = (res > 255) ? 255 : res;
			src[y * cdim + x] = (unsigned char)res;
		}
}
