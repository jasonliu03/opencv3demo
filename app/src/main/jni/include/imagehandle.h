#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int DetectThreshold(IplImage*src);
Mat Binary(Mat imgInput);
Mat sobel(Mat inputImage);
Mat RGB2HSV(Mat inputImage);
Mat RGB2Lab(Mat inputImage);
IplImage* DetectRelief(IplImage* input);
Mat Relief(Mat imgInput);
Mat sketch(Mat imgInput);
Mat sumiao(Mat Image_in);

Mat tongueDivision(Mat tongueROI);

