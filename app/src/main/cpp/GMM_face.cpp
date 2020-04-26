#include "opencv2/opencv.hpp"
#include "GMM.h"

using namespace std;
using namespace cv;


void cdbt(const IplImage* mask, const IplImage* grayImage, int& i_dark,
          int& i_bright, double rate_dark, double rate_bright) {
  float data[256] = {0};
  int total = 0;
  for (int y = 0; y < mask->height; y++) {
    for (int x = 0; x < mask->width; x++) {
      if (cvGetReal2D(mask, y, x) > 200) {
        int intensity = cvRound(cvGetReal2D(grayImage, y, x));
        data[intensity]++;
        total++;
      }
    }
  }

  float sum_dark = 0;
  for (int i = 0; i < 256; i++) {
    sum_dark += data[i];
    if ((sum_dark) / ((double)total) > rate_dark) {
      i_dark = i;
      break;
    }
  }
  float sum_bright = 0;
  for (int i = 255; i >= 0; i--) {
    sum_bright += data[i];
    if ((sum_bright) / ((double)total) > rate_bright) {
      i_bright = i;
      break;
    }
  }
}

GMM* bbmm(IplImage *faceImage) {
  int len;
  len = (faceImage->height > faceImage->width) ? faceImage->height
                                               : faceImage->width;
  double type = ((double)len) / 50.0;
  IplImage* faceImageScale =
      cvCreateImage(cvSize(cvRound(faceImage->width / type),
                           cvRound(faceImage->height / type)),
                    faceImage->depth, faceImage->nChannels);
  cvResize(faceImage, faceImageScale);
  IplImage* faceImageScaleGray = cvCreateImage(cvGetSize(faceImageScale), 8, 1);
  cvCvtColor(faceImageScale, faceImageScaleGray, CV_BGR2GRAY);
//  cvSmooth(faceImageScaleGray, faceImageScaleGray, CV_GAUSSIAN, 5, 5);
  IplImage* faceScaleEllipseMask =
      cvCreateImage(cvGetSize(faceImageScale), 8, 1);
  cvZero(faceScaleEllipseMask);
  CvPoint center = cvPoint(cvRound(faceScaleEllipseMask->width * 0.5),
                           cvRound(faceScaleEllipseMask->height * 0.5));
  CvSize size = cvSize(cvRound(faceScaleEllipseMask->width * 0.36),
                       cvRound(faceScaleEllipseMask->height * 0.50));
  cvEllipse(faceScaleEllipseMask, center, size, 0, 0, 360, cvScalar(255),
            CV_FILLED);
  int i_dark, i_bright;
  double rate_dark = 0.2, rate_bright = 0;
  cdbt(faceScaleEllipseMask, faceImageScaleGray, i_dark, i_bright, rate_dark,
       rate_bright);
  for (int y = 0; y < faceScaleEllipseMask->height; ++y) {
    for (int x = 0; x < faceScaleEllipseMask->width; ++x) {
      if (cvGetReal2D(faceImageScaleGray, y, x) < i_dark) {
        cvSetReal2D(faceScaleEllipseMask, y, x, 0);
      }
    }
  }
  IplImage* faceImageScale_Lab = NULL;
  faceImageScale_Lab = cvCreateImage(cvGetSize(faceImageScale), 8, 3);
  cvCvtColor(faceImageScale, faceImageScale_Lab, CV_BGR2Lab);
  uint cnt = 0, nrows = 0;
  for (int y = 0; y < faceScaleEllipseMask->height; y++) {
    for (int x = 0; x < faceScaleEllipseMask->width; x++) {
      if (cvGetReal2D(faceScaleEllipseMask, y, x) > 200) {
        nrows++;
      }
    }
  }
  double** data;
  data = (double**)malloc(nrows * sizeof(double*));
  for (int i = 0; i < nrows; i++) data[i] = (double*)malloc(3 * sizeof(double));
  for (int y = 0; y < faceScaleEllipseMask->height; y++) {
    for (int x = 0; x < faceScaleEllipseMask->width; x++) {
      if (cvGetReal2D(faceScaleEllipseMask, y, x) > 200) {
        data[cnt][0] = cvGet2D(faceImageScale_Lab, y, x).val[0];
        data[cnt][1] = cvGet2D(faceImageScale_Lab, y, x).val[1];
        data[cnt++][2] = cvGet2D(faceImageScale_Lab, y, x).val[2];
      }
    }
  }
  GMM* mComplexionGMM = NULL;
  int nGMM = 3;
  mComplexionGMM = new GMM(nGMM);
  mComplexionGMM->Build(data, nrows);
  for (int i = 0; i < nrows; i++) free(data[i]);
  free(data);
  cvReleaseImage(&faceImageScale);
  cvReleaseImage(&faceImageScaleGray);
  cvReleaseImage(&faceScaleEllipseMask);
  cvReleaseImage(&faceImageScale_Lab);
  return mComplexionGMM;
}

IplImage* ccpp(IplImage *faceImage, GMM *mComplexionGMM) {
  IplImage *faceImage_Lab = cvCreateImage(cvGetSize(faceImage), 8, 3);
  cvCvtColor(faceImage, faceImage_Lab, CV_BGR2Lab);

  IplImage *faceImage_Gauss = cvCreateImage(cvGetSize(faceImage), IPL_DEPTH_64F, 1);
  cvZero(faceImage_Gauss);
  for (int y = 0; y < faceImage_Gauss->height; y++) {
    for (int x = 0; x < faceImage_Gauss->width; x++) {
      CvScalar pixel = cvGet2D(faceImage_Lab, y, x);
      Color c(pixel.val[0], pixel.val[1], pixel.val[2]);
      float px = mComplexionGMM->p(c);
      cvSetReal2D(faceImage_Gauss, y, x, px);
    }
  }
  cvNormalize(faceImage_Gauss, faceImage_Gauss, 1.0, 0.0, CV_C);
  cvSmooth(faceImage_Gauss, faceImage_Gauss, CV_GAUSSIAN, 5, 5);

  IplImage *faceImage_GaussRange = cvCreateImage(cvGetSize(faceImage), 8, 1);
  cvZero(faceImage_GaussRange);
  for (int y = 0; y < faceImage_Gauss->height; y++) {
    for (int x = 0; x < faceImage_Gauss->width; x++) {
      cvSetReal2D(faceImage_GaussRange, y, x, (int)(cvGetReal2D(faceImage_Gauss, y, x)*255));
    }
  }
  cvReleaseImage(&faceImage_Lab);
  cvReleaseImage(&faceImage_Gauss);
  delete mComplexionGMM;
  return faceImage_GaussRange;
}
