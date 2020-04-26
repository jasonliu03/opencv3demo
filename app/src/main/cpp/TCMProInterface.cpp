#include "TCMProInterface.h"
#include "TypeConvert.h"
#include "tongueDiagnosis.h"
#include <stdlib.h>
#include <crtdbg.h>
#include<algorithm>
#include<sstream>

void lightAdjust(Mat& img, Mat& cImg, float ga)
{
	if (cImg.empty())
		cImg.create(img.rows, img.cols, img.type());

	int i, j;
	Size size = img.size();
	int chns = img.channels();

	if (img.isContinuous() && cImg.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}

	ga = ga / 10.0;
	if (ga<0.1) ga = -0.1;
	if (ga> 5.0) ga = 5.0;

	unsigned char lut[256];
	for (i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(cv::pow((float)(i / 255.0), ga) * 255.0f);
	}

	for (i = 0; i<size.height; ++i)
	{
		const unsigned char* src = (const unsigned char*)(img.data + img.step*i);
		unsigned char* dst = (unsigned char*)cImg.data + cImg.step*i;
		for (j = 0; j<size.width; ++j)
		{
			dst[j*chns] = lut[src[j*chns]];
			dst[j*chns + 1] = lut[src[j*chns + 1]];
			dst[j*chns + 2] = lut[src[j*chns + 2]];
		}
	}
}

IplImage* promoteV(const IplImage* Pimage_in1) {
	IplImage *hsv = cvCreateImage(cvGetSize(Pimage_in1), 8, 3);
	cvCvtColor(Pimage_in1, hsv, CV_BGR2HSV);
	std::vector<cv::Mat>channels;
	split(hsv, channels);
	int vMin = 255;
	int vMax = 0;
	for (int i = 0; i < channels[2].rows; i++)
	{
		uchar *pImg = channels[2].ptr<uchar>(i);
		for (int j = 0; j < channels[2].cols; j++)
		{
			int v = (int)pImg[j];
			if (v > vMax) vMax = v;
			if (v < vMin) vMin = v;
		}
	}
	float Max = vMax;
	float Min = vMin;
	float M = 200;
	float r = M / (vMax - vMin);
	for (int m = 0; m < channels[2].rows; m++)
	{
		uchar *pI = channels[2].ptr<uchar>(m);
		for (int n = 0; n < channels[2].cols; n++)
		{
			double newPixel = ((int)pI[n] - vMin) * r;
			if (newPixel > 255) pI[n] = 255;
			else if (newPixel < 0) pI[n] = 0;
			else
				pI[n] = (uchar)newPixel;
		}
	}
	cv::Mat mergeImg;
	cv::merge(channels, mergeImg);
	IplImage imgTmp = mergeImg;
	IplImage *input = cvCloneImage(&imgTmp);
	IplImage *image = cvCreateImage(cvGetSize(input), 8, 3);
	cvCvtColor(input, image, CV_HSV2BGR);
	cvReleaseImage(&hsv);
	cvReleaseImage(&input);
	return image;
}

IplImage* preprocessLightAdjust(const IplImage* Pimage_in, int haarFaceDetectSwitch) {
	int faceDetectRes = 0;
	IplImage* Pimage_out = NULL;
	IplImage* Pimage_in2 = cvCloneImage(Pimage_in);
	fsdmm(Pimage_in2, haarFaceDetectSwitch);
	bool isOk2 = fsdbmpf();
	if (isOk2) {
		faceDetectRes = 1;
		IplImage* ffii = gefit();
		IplImage* fsi = gfsi();
		IplImage *OverHsv = cvCreateImage(cvGetSize(fsi), 8, 3);
		cvCvtColor(fsi, OverHsv, CV_BGR2HSV);
		std::vector<cv::Mat>channels;
		split(OverHsv, channels);
		float count_V = 0;
		float threshold = 190;
		float count_D = 0;
		float thresholdD = 130;
		for (int i = 0; i < channels[2].rows; i++)
		{
			uchar *pImg = channels[2].ptr<uchar>(i);
			for (int j = 0; j < channels[2].cols; j++)
			{
				int v = (int)pImg[j];
				if (v > threshold && v < 255)  count_V += 1;
				if (v < thresholdD) count_D += 1;
			}
		}

		float V_ratio = count_V / (channels[2].rows * channels[2].cols);

		float D_ratio = count_D / (channels[2].rows * channels[2].cols);
		if (V_ratio > 0.25) {
			Pimage_out = promoteV(Pimage_in2);
		}
		else if (D_ratio > 0.3) {
			int g;
			if (D_ratio > 0.3 && D_ratio < 0.4) {
				g = 6;
			}
			else {
				g = 5;
			}
			Mat ImgTmp;
			ImgTmp = cvarrToMat(Pimage_in2);
			Mat cImg;
			lightAdjust(ImgTmp, cImg, g);
			IplImage *image_input = &IplImage(cImg);
			Pimage_out = cvCloneImage(image_input);
		}
		else {
			Pimage_out = cvCloneImage(Pimage_in2);
		}
		cvReleaseImage(&OverHsv);
	}
	else {
		Pimage_out = cvCloneImage(Pimage_in2);
	}
	cvReleaseImage(&Pimage_in2);
	return Pimage_out;
}

char* tcmFacePro(IplImage* input, int haarFaceDetectSwitch /*=0*/) {
  string faceDagnosisDataSet = "";
  int faceDetectRes = 0;
  int faceColor = 0;
  int faceGloss = 0;
  int lipDetectRes = 0;
  int lipColor = 0;
  IplImage* image_in = cvCloneImage(input);
  //IplImage* Pimage_in3 = cvCloneImage(input);
  //IplImage* image_in = preprocessLightAdjust(Pimage_in3,haarFaceDetectSwitch);
  fsdmm(image_in, haarFaceDetectSwitch);
  bool isOk = fsdbmpf();
  if (isOk) {
    faceDetectRes = 1;
    IplImage* ffii = gefit();
    IplImage* fsi = gfsi();
    CvScalar faceSkinColorFeature = efscf();
    faceDagnosisDataSet += toString(cvRound(faceSkinColorFeature.val[0])) +
                           "," +
                           toString(cvRound(faceSkinColorFeature.val[1])) +
                           "," + toString(cvRound(faceSkinColorFeature.val[2]));
    faceColor = copdt(faceSkinColorFeature);
    CvScalar faceGlossFeature = glopt(ffii);
    faceGloss = cvRound(faceGlossFeature.val[0]);
    faceDagnosisDataSet += "," + toString(faceGlossFeature.val[1]) + "," +
                           toString(faceGlossFeature.val[2]) + "," +
                           toString(faceGlossFeature.val[3]);
    lsgi(ffii);
    bool isLip = prodp();
    if (isLip) {
      lipDetectRes = 1;
      IplImage* lpei = gtli();
      CvScalar lipColorFeature = elcf();
      faceDagnosisDataSet += "," + toString(cvRound(lipColorFeature.val[0])) +
                             "," + toString(cvRound(lipColorFeature.val[1])) +
                             "," + toString(cvRound(lipColorFeature.val[2]));
      lipColor = cptz(lipColorFeature);
    } else {
      lipDetectRes = 0;
      faceDagnosisDataSet += "";
    }
  } else {
    faceDetectRes = 0;
    faceDagnosisDataSet = "";
  }
  int result = 0;
  result = faceDetectRes * 1 + faceColor * 10 + faceGloss * 100 +
           lipDetectRes * 1000 + lipColor * 10000;
  string resultStr = toString(result);
  static string tcmFaceFeature = "";
  tcmFaceFeature = resultStr + "," + faceDagnosisDataSet;
  cvReleaseImage(&image_in);
  //cvReleaseImage(&Pimage_in3);
  return const_cast<char*>(tcmFaceFeature.c_str());
}

char* tcmTonguePro(IplImage* input, int haarTongueDetectSwitch /*=0*/) {
  string tongueDagnosisDataSet = "";
  int tongueDetectRes = 0;
  int tongueCrack = 0;
  int tongueFatThin = 0;
  int tongueCoatThickness = 0;
  int tongueCoatColor = 0;
  int tongueNatureColor = 0;
  IplImage* image_in = cvCloneImage(input);
  ttssii(image_in, haarTongueDetectSwitch);
  bool isOk = ppff();
  if (isOk) {
    tongueDetectRes = 1;
    IplImage* mmm = gtmm();
    IplImage* ttt = gtii();
    IplImage* tongueMIR = gtim();
    int nCracks = tcdd(tongueMIR);
    tongueDagnosisDataSet = toString(nCracks);
    if (nCracks > 0) {
      tongueCrack = 1;
    } else if (nCracks == 0) {
      tongueCrack = 0;
    }
    tftd();
    tongueFatThin = prdt(mmm);
    tcns(ttt, mmm);
    pffl();
    IplImage* tci = gtci();
    IplImage* tni = gtni();
    tctd(tci, tni);
    tongueCoatThickness = prbh();
    CvScalar tongueNatureColorFeature = tsbd(ttt, tongueNatureColor);
    tongueDagnosisDataSet +=
        "," + toString(cvRound(tongueNatureColorFeature.val[0])) + "," +
        toString(cvRound(tongueNatureColorFeature.val[1])) + "," +
        toString(cvRound(tongueNatureColorFeature.val[2]));
    tccr(tci);
    CvScalar tongueCoatColorFeature = etccf();
    tongueDagnosisDataSet +=
        "," + toString(cvRound(tongueCoatColorFeature.val[0])) + "," +
        toString(cvRound(tongueCoatColorFeature.val[1])) + "," +
        toString(cvRound(tongueCoatColorFeature.val[2]));
    tongueCoatColor = ccopd(tongueCoatColorFeature);
  } else {
    tongueDetectRes = 0;
    tongueDagnosisDataSet = "";
  }
  int result = 0;
  result = tongueDetectRes * 1 + tongueCrack * 10 + tongueFatThin * 100 +
           tongueCoatThickness * 1000 + tongueCoatColor * 10000 +
           tongueNatureColor * 100000;
  string resultStr = toString(result);
  static string tcmTongueFeature = "";
  tcmTongueFeature = resultStr + "," + tongueDagnosisDataSet;
  cvReleaseImage(&image_in);
  return const_cast<char*>(tcmTongueFeature.c_str());
}

