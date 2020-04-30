#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat img, inpaintMask;
Mat res;
Point prevPt(-1, -1);

//onMouse function
static void onMouse(int event, int x, int y, int flags, void*) {
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(img, prevPt, pt, Scalar(0, 255, 0), 5, 8, 0);
		prevPt = pt;
		imshow("image", img);
		imshow("image: mask", inpaintMask);
	}
}

int main() {
	string filename = "sample.png";
	img = imread(filename, IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load image: " << filename << endl;
		return 0;
	}

	Mat imgCopy = img.clone();
	inpaintMask = Mat::zeros(imgCopy.size(), CV_8U);

	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", img);
	imshow("image: mask", inpaintMask);
	setMouseCallback("image", onMouse, NULL);

	for (;;) {
		char c = (char)waitKey();
		if (c == 't') {
			inpaint(imgCopy, inpaintMask, res, 3, INPAINT_TELEA);
			imshow("Inpaint Output using FMM", res);
		}
		if (c == 'n') {
			inpaint(imgCopy, inpaintMask, res, 3, INPAINT_NS);
			imshow("Inpaint output using NS Technique", res);
		}
		if (c == 'r') {
			inpaintMask = Scalar::all(0);
			imgCopy.copyTo(img);
			imshow("image", img);
			imshow("image: mask", inpaintMask);
		}
		if (c == 27)
			break;
	}

	destroyAllWindows();

	return 0;
}

