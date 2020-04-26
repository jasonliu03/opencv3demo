#include "RemoveNoise.h"

RemoveNoise::RemoveNoise() {}
RemoveNoise::~RemoveNoise() {}
void RemoveNoise::LessConnectedRegionRemove(IplImage* image, int area) {
  IplImage* src = cvCloneImage(image);
  CvMemStorage* storage = cvCreateMemStorage(0);
  CvSeq* contour = 0;
  cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE);
  cvZero(image);
  double minarea = (double)area;
  for (; contour != 0; contour = contour->h_next) {
    double tmparea = fabs(cvContourArea(contour));
    if (tmparea < minarea) {
      cvSeqRemove(contour, 0);
      continue;
    }
    CvScalar color = CV_RGB(255, 255, 255);
    cvDrawContours(image, contour, color, color, -1, CV_FILLED, 8,
                   cvPoint(0, 0));
  }
  cvReleaseImage(&src);
  cvReleaseMemStorage(&storage);
}
int RemoveNoise::RemoveCrackImageNoise(IplImage* image, int area,
                                       int lineLength) {
  IplImage* src = cvCloneImage(image);
  CvMemStorage* storage = cvCreateMemStorage(0);
  CvSeq* contour = 0;
  cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE);
  cvZero(image);
  double minarea = (double)area;
  int nCracks = 0;
  for (; contour != 0; contour = contour->h_next) {
    CvRect rect = cvBoundingRect(contour);
    int len = rect.height > rect.width ? rect.height : rect.width;
    double tmparea = fabs(cvContourArea(contour));
    if (tmparea < minarea || len < lineLength) {
      cvSeqRemove(contour, 0);
      continue;
    }
    nCracks++;
    CvScalar color = CV_RGB(255, 255, 255);
    cvDrawContours(image, contour, color, color, -1, CV_FILLED, 8,
                   cvPoint(0, 0));
  }
  cvReleaseImage(&src);
  cvReleaseMemStorage(&storage);
  return nCracks;
}
