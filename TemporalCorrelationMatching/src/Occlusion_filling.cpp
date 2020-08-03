#include "Occlusion_filling.h"


void OcclusionFilling(Mat Unfill) {

	for (int i = 0; i < Unfill.rows; i++) {
		for (int j = 0; j < Unfill.cols; j++) {
			if (Unfill.ptr<uint>(i)[j] == 0) {
				uint grayvalue1;
				uint grayvalue2;
				if (j == 0) {
					for (int n = j+1; n <= Unfill.cols; n++) {
						if (Unfill.ptr<uint>(i)[n] != 0) {
							grayvalue1 = Unfill.ptr<uint>(i)[n];
							break;
						}
					}
					Unfill.ptr<uint>(i)[j] = grayvalue1;
				}
				else {
					for (int n = j+1; n <= Unfill.cols; n++) {
						if (Unfill.ptr<uint>(i)[n] != 0) {
							grayvalue1 = Unfill.ptr<uint>(i)[n];
							break;
						}
					}
					for (int m = j - 1; m >= 0; m--) {
						if (Unfill.ptr<uint>(i)[m] != 0) {
							grayvalue2 = Unfill.ptr<uint>(i)[m];
							break;
						}
					}
					Unfill.ptr<uint>(i)[j] = min(grayvalue1, grayvalue2);
				}
			}
		}
	}

}