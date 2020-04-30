/*
 * Blemish remover is a point and click clone tool that selects the smoothest patch near to the clicked area and pastes it over top.
 * Completed for OpenCV's Computer Vision 1 course.
 * Author: Richard Purcell April 30, 2020.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat fg, bg, mask, dst, paddedDst;
Point center(-1,-1);
const int cloneSize = 40;
const int rangeSize = 8;
cv::Range rangesX[rangeSize];
cv::Range rangesY[rangeSize];

static void onMouse(int event, int x, int y, int flags, void*);
int minLaplacian();
void setRanges();
void makeMask();
void drawRanges();


int main() {
	string bg_filename = "./blemish.png";
	bg = imread(bg_filename, IMREAD_COLOR);
	mask = 255 * Mat::zeros(cloneSize, cloneSize, CV_8U);
	dst = bg.clone();
	makeMask();
	namedWindow("clone", WINDOW_AUTOSIZE);
	imshow("clone", dst);
	setMouseCallback("clone", onMouse, NULL);
	cout << "press r to reset" << endl;
	cout << "press esc to exit" << endl;
	for (;;) {
		char c = (char)waitKey();
		if (c == 'r') {
			dst = bg.clone();
			imshow("clone", dst);
		}
		if (c == 27)
			break;
	}

	destroyAllWindows();

	return 0;
}

/*
* The onMouse function launches other functions such as setRanges(), minLaplacian(),
* and seamlessClone() to complete the destination image.
*/
static void onMouse(int event, int x, int y, int flags, void*) {
	if (event == EVENT_LBUTTONDOWN) {
		//set the pixel where mouse has been clicked
		center = Point(x + 2*cloneSize, y + 2*cloneSize);
		//pad the background image
		copyMakeBorder(dst, paddedDst, 2*cloneSize, 2*cloneSize, 2*cloneSize, 2*cloneSize, BORDER_REFLECT);
		//set the ranges for patches to test
		setRanges();
		//get patch with least noise
		int minPatchIndex = minLaplacian();
		fg = paddedDst(Rect(Point(rangesX[minPatchIndex].start, rangesY[minPatchIndex].start),
			Point(rangesX[minPatchIndex].end, rangesY[minPatchIndex].end)));
		//clone
		seamlessClone(fg, paddedDst, mask, center, paddedDst, MONOCHROME_TRANSFER);
		//crop
		dst = paddedDst(Range(2*cloneSize, 2*cloneSize + bg.size().height), Range(2*cloneSize, 2*cloneSize + bg.size().width));

		imshow("clone", dst);
	}
}

/*
* The minLaplacian function returns the index of the smoothest patch surrounding the selected patch.
*/
int minLaplacian() {
	double min = 10000.0;
	double minTemp;
	int minIndex = 0;
	Mat patch, laplacianPatch;
	Scalar roiMean = (0, 0, 0);

	for (int i = 0; i < rangeSize; i++) {
		minTemp = 0.0;
		patch = paddedDst(Rect(Point(rangesX[i].start, rangesY[i].start),
			Point(rangesX[i].end, rangesY[i].end)));
		cvtColor(patch, patch, COLOR_BGR2GRAY);

		Laplacian(patch, laplacianPatch, CV_64FC1, 5, .001);

		roiMean = mean(laplacianPatch);

		for (int h_i = 0; h_i < patch.rows; h_i++) {
			for (int w_i = 0; w_i < patch.cols; w_i++) {
				minTemp += pow(abs(laplacianPatch.at<double>(h_i, w_i)) - roiMean(0), 2);
			}
		}

		if (minTemp < min) {
			min = minTemp;
			minIndex = i;
		}
	}
	return minIndex;
}

/*
* The setRanges function sets the pixel values contained in the 8 patches
* that surround the selected patch.
*/
void setRanges() {

	rangesX[0] = Range((center.x - (3 * cloneSize) / 2), center.x - cloneSize / 2);
	rangesY[0] = Range((center.y - (3 * cloneSize) / 2), center.y - cloneSize / 2);

	rangesX[1] = Range((center.x - cloneSize / 2), center.x + cloneSize / 2);
	rangesY[1] = Range((center.y - (3 * cloneSize) / 2), center.y - cloneSize / 2);

	rangesX[2] = Range((center.x + cloneSize / 2), center.x + (3 * cloneSize) / 2);
	rangesY[2] = Range((center.y - (3 * cloneSize) / 2), center.y - cloneSize / 2);

	rangesX[3] = Range((center.x - (3 * cloneSize) / 2), center.x - cloneSize / 2);
	rangesY[3] = Range((center.y - cloneSize / 2), center.y + cloneSize / 2);

	rangesX[4] = Range((center.x + cloneSize / 2), center.x + (3 * cloneSize) / 2);
	rangesY[4] = Range((center.y - cloneSize / 2), center.y + cloneSize / 2);

	rangesX[5] = Range((center.x - (3 * cloneSize) / 2), center.x - cloneSize / 2);
	rangesY[5] = Range((center.y + cloneSize / 2), center.y + (3 * cloneSize) / 2);

	rangesX[6] = Range((center.x - cloneSize / 2), center.x + cloneSize / 2);
	rangesY[6] = Range((center.y + cloneSize / 2), center.y + (3 * cloneSize) / 2);

	rangesX[7] = Range((center.x + cloneSize / 2), center.x + (3 * cloneSize) / 2);
	rangesY[7] = Range((center.y + cloneSize / 2), center.y + (3 * cloneSize) / 2);
}

/*
* The makeMask function creates a circular mask with soft edges.
*/
void makeMask() {
	int blursize = 0;
	circle(mask, Point(mask.cols/2, mask.rows/2), cloneSize / 3, Scalar(255, 255, 255), -1);
	if ((cloneSize/2) % 2 == 0)
		blursize = cloneSize / 2 -1;
	else
		blursize = cloneSize / 2;

	GaussianBlur(mask, mask, Size(blursize, blursize), false, mask.depth());
}

/*
* The drawRanges function creates a graphic representation of the selected patch
* and the 8 patches that surround it.
*/
void drawRanges() {
	Mat rangeOutlines;
	rangeOutlines = paddedDst.clone();
	circle(rangeOutlines, center, 10, Scalar(0, 255, 0), 3);
	for (int i = 0; i < rangeSize; i++) {
		rectangle(rangeOutlines,
			Point(rangesX[i].start, rangesY[i].start),
			Point(rangesX[i].end, rangesY[i].end),
			Scalar(255, 255, 255), 1);
	}
	imshow("rangeOutlines", rangeOutlines);
}


