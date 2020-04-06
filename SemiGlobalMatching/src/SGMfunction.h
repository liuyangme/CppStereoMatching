#ifndef __SGMSTEREO_H__
#define __SGMSTEREO_H__

#include<bitset>
#define CENSUS_BIT 80
using namespace cv;


class SGMStereo {
public:
	SGMStereo();

	void setDisparityTotal(const int disparityTotal);
	void setDisparityFactor(const double disparityFactor);
	void setDataCostParameters(const int sobelCapValue,
		const int censusWindowRadius,
		const double censusWeightFactor,
		const int aggregationWindowRadius);
	void setSmoothnessCostParameters(const int smoothnessPenaltySmall, const int smoothnessPenaltyLarge);
	void setConsistencyThreshold(const int consistencyThreshold);

	void compute(Mat& leftImage, Mat& rightImage, Mat& disparityImage);

private:
	void initialize(const Mat& leftImage, const Mat& rightImage);
	void setImageSize(const Mat& leftImage, const Mat& rightImage);
	void allocateDataBuffer();
	void freeDataBuffer();
	void computeCostImage(Mat& leftImage, Mat& rightImage);
	void convertToGrayscale(Mat& leftImage,
		Mat& rightImage,
		unsigned char* leftGrayscaleImage,
		unsigned char* rightGrayscaleImage) ;
	void computeLeftCostImage(const unsigned char* leftGrayscaleImage, const unsigned char* rightGrayscaleImage);
	void computeCappedSobelImage(const unsigned char* image, const bool horizontalFlip, unsigned char* sobelImage) const;
	void computeCensusImage();
	void calcTopRowCost(unsigned char*& leftSobelRow, std::bitset<CENSUS_BIT>*& leftCensusRow,
		unsigned char*& rightSobelRow, std::bitset<CENSUS_BIT>*& rightCensusRow,
		unsigned short* costImageRow);
	void calcRowCosts(unsigned char*& leftSobelRow, std::bitset<CENSUS_BIT>*& leftCensusRow,
		unsigned char*& rightSobelRow, std::bitset<CENSUS_BIT>*& rightCensusRow,
		unsigned short* costImageRow);
	void calcPixelwiseSAD(const unsigned char* leftSobelRow, const unsigned char* rightSobelRow);
	void calcHalfPixelRight(const unsigned char* rightSobelRow);
	void addPixelwiseHamming(const std::bitset<CENSUS_BIT>* leftCensusRow, const std::bitset<CENSUS_BIT>* rightCensusRow);
	void computeRightCostImage();
	void performSGM(unsigned short* costImage, unsigned short* disparityImage);
	void speckleFilter(const int maxSpeckleSize, const int maxDifference, unsigned short* image) const;
	void enforceLeftRightConsistency(unsigned short* leftDisparityImage, unsigned short* rightDisparityImage) const;


	// Parameter
	int disparityTotal_;
	double disparityFactor_;
	int sobelCapValue_;
	int censusWindowRadius_;
	double censusWeightFactor_;
	int aggregationWindowRadius_;
	int smoothnessPenaltySmall_;
	int smoothnessPenaltyLarge_;
	int consistencyThreshold_;

	// Data
	int width_;
	int height_;
	int widthStep_;
	unsigned short* leftCostImage_;
	unsigned short* rightCostImage_;
	std::bitset<CENSUS_BIT>* lCode;
	std::bitset<CENSUS_BIT>* rCode;
	unsigned char* pixelwiseCostRow_;
	unsigned short* rowAggregatedCost_;
	unsigned char* halfPixelRightMin_;
	unsigned char* halfPixelRightMax_;
	int pathRowBufferTotal_;
	int disparitySize_;
	int pathTotal_;
	int pathDisparitySize_;
	int costSumBufferRowSize_;
	int costSumBufferSize_;
	int pathMinCostBufferSize_;
	int pathCostBufferSize_;
	int totalBufferSize_;
	short* sgmBuffer_;
	Mat lGray;
	Mat rGray;
};


#endif
