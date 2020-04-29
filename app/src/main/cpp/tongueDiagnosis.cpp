#include "tongueDiagnosis.h"
#include "RemoveNoise.h"
#include "GMM_face.h"
CvMemStorage* storageS;
CvHaarClassifierCascade* cascadeS;
char const* cascade_nameS = "haarcascade_tongue.xml";
IplImage* imageSrcS;
IplImage* inputImageS;
IplImage* image_tongueROI;
IplImage* image_v;
IplImage* image_vBiPolar;
IplImage* image_vBiPolarcftt;
IplImage* image_vMask;
IplImage* image_tongue;
CvRect tongueRect;
bool isDetectTongue;
int switchHaarTongue;
IplImage* image_MIRTongue;
IplImage* I;
IplImage* p;
int rlvbo;
double eps;
int N;
int rcucao;
IplImage* win_mat;
IplImage* inputImagecucao;
IplImage* image_sdv;
IplImage* image_roughness;
IplImage* q;
int rdu;
IplImage* inputImageandu;
IplImage* darkImageBi;
int T;
int g;
IplImage* inputImageshuiliu;
IplImage* inputRoughness;
IplImage* L;
char const* panClassifierPathName = "svm_tonguefatthin.xml";
IplImage* tongueImage;
IplImage* tongueMask;
IplImage* tongueCoat;
IplImage* tongueNature;
IplImage* tongueCoatImage;
IplImage* tongueNatureImage;
float feature[2];
char const* bohouClassifierPathName = "svm_tonguecoatthickness.xml";
IplImage* tongueCoatImagecc;
char const* ccolorClassifierPathName = "svm_tonguecoatcolor.xml";
double roiScale = 1.0;

#define cvQueryHistValue_1D( hist, idx0 ) \
    ((float)cvGetReal1D( (hist)->bins, (idx0)))

void tccr(const IplImage* tongueCoatImage1) {
  tongueCoatImagecc = cvCloneImage(tongueCoatImage1);
}
int ccopd(const CvScalar coatColor) {
  int response = 0;
  float feature[2];
  feature[0] = coatColor.val[1];
  feature[1] = coatColor.val[2];
  Mat testDataMat(1, 2, CV_32FC1, feature);
  //CvSVM svm = CvSVM();
//  CvSVM svm;
//  svm.load(ccolorClassifierPathName);
//  response = (int)svm.predict(testDataMat);

  Mat responses;
  Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(ccolorClassifierPathName); //读取模型
  svm->predict(testDataMat, responses);
  responses.convertTo(responses,CV_32S);
  response = responses.at<int>(0,0);
  return response;
}
CvScalar etccf() {
  CvRect coatRect =
      cvRect(tongueCoatImagecc->width / 5, tongueCoatImagecc->height / 5,
             tongueCoatImagecc->width * 3 / 5, tongueCoatImagecc->height / 2);
  cvSetImageROI(tongueCoatImagecc, coatRect);
  IplImage* tongueCoatRegion23 =
      cvCreateImage(cvGetSize(tongueCoatImagecc), 8, 3);
  cvCopy(tongueCoatImagecc, tongueCoatRegion23);
  cvResetImageROI(tongueCoatImagecc);
  IplImage* tongueCoatRegionMask23 =
      cvCreateImage(cvGetSize(tongueCoatRegion23), 8, 1);
  cvCvtColor(tongueCoatRegion23, tongueCoatRegionMask23, CV_BGR2GRAY);
  cvThreshold(tongueCoatRegionMask23, tongueCoatRegionMask23, 254, 255,
              CV_THRESH_BINARY_INV);
  IplImage* tongueCoatRegion23_Lab =
      cvCreateImage(cvGetSize(tongueCoatRegion23), 8, 3);
  cvCvtColor(tongueCoatRegion23, tongueCoatRegion23_Lab, CV_BGR2Lab);
  CvScalar tongueCoatColorFeature =
      cvAvg(tongueCoatRegion23_Lab, tongueCoatRegionMask23);
  cvReleaseImage(&tongueCoatRegion23);
  cvReleaseImage(&tongueCoatRegionMask23);
  cvReleaseImage(&tongueCoatRegion23_Lab);
  return tongueCoatColorFeature;
}
void exfe() {
  CvRect coatRect =
      cvRect(tongueCoatImage->width / 5, 2, tongueCoatImage->width * 3 / 5,
             tongueCoatImage->height * 3 / 5);
  CvScalar tongueCoatFeature;
  tongueCoatFeature = tofe(tongueCoatImage);
  CvScalar tongueNatureFeature;
  tongueNatureFeature = tofe(tongueNatureImage);
  float distance_a = abs(tongueCoatFeature.val[0] - tongueNatureFeature.val[0]);
  float distance_b = abs(tongueCoatFeature.val[1] - tongueNatureFeature.val[1]);
  feature[0] = max(distance_a, distance_b);
  feature[1] = tcdr(tongueCoatImage, tongueNatureImage, tongueCoatFeature,
                    tongueNatureFeature, coatRect);
}
CvScalar tofe(IplImage* image) {
  CvScalar tongueFeature = cvScalar(0);
  IplImage* image_Lab = NULL;
  IplImage* image_L = NULL;
  IplImage* image_A = NULL;
  IplImage* image_B = NULL;
  image_Lab = cvCreateImage(cvGetSize(image), 8, 3);
  image_L = cvCreateImage(cvGetSize(image), 8, 1);
  image_A = cvCreateImage(cvGetSize(image), 8, 1);
  image_B = cvCreateImage(cvGetSize(image), 8, 1);

  cvCvtColor(image, image_Lab, CV_BGR2Lab);
  cvSplit(image_Lab, image_L, image_A, image_B, NULL);

  for (int y = 0; y < image_L->height; y++) {
    for (int x = 0; x < image_L->width; x++) {
      if (cvGetReal2D(image_L, y, x) == 255) {
        cvSetReal2D(image_A, y, x, 255);
        cvSetReal2D(image_B, y, x, 255);
      }
    }
  }
  int a = cohi(image_A);
  int b = cohi(image_B);
  tongueFeature.val[0] = a;
  tongueFeature.val[1] = b;

  cvReleaseImage(&image_Lab);
  cvReleaseImage(&image_L);
  cvReleaseImage(&image_A);
  cvReleaseImage(&image_B);

  return tongueFeature;
}
int cohi(IplImage* src) {
  IplImage* gray_plane = cvCreateImage(cvGetSize(src), 8, 1);
  if (src->nChannels == 3) {
    cvCvtColor(src, gray_plane, CV_BGR2GRAY);
  } else {
    cvCopy(src, gray_plane);
  }

  int hist_size = 256;
  float range[] = {0, 255};
  float* ranges[] = {range};
  CvHistogram* gray_hist =
      cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);

  cvCalcHist(&gray_plane, gray_hist, 0, 0);

  cvNormalizeHist(gray_hist, 1.0);

  int scale = 2;
  int hist_height = 256;
  IplImage* hist_image =
      cvCreateImage(cvSize(hist_size * scale, hist_height), 8, 3);
  cvZero(hist_image);
  float max_value = 0;
  cvGetMinMaxHistValue(gray_hist, 0, &max_value, 0, 0);

  int max_id;
  int intensity = 0;
  int temp = 0;
  for (int i = 0; i < hist_size; i++) {
    float bin_val = cvQueryHistValue_1D(gray_hist, i);
    intensity = cvRound(bin_val * hist_height / max_value);
    if (intensity > temp && i > 10 && i < hist_size - 1) {
      max_id = i;
      temp = intensity;
    }
    cvRectangle(hist_image, cvPoint(i * scale, hist_height - 1),
                cvPoint((i + 1) * scale - 1, hist_height - intensity),
                CV_RGB(255, 255, 255));
  }

  float bin_val = cvQueryHistValue_1D(gray_hist, max_id);
  intensity = cvRound(bin_val * hist_height / max_value);
  cvRectangle(hist_image, cvPoint(max_id * scale, hist_height - 1),
              cvPoint((max_id + 1) * scale - 1, hist_height - intensity),
              CV_RGB(255, 0, 0));

  cvReleaseImage(&gray_plane);
  cvReleaseImage(&hist_image);
  cvReleaseHist(&gray_hist);
  return max_id;
}
float tcdr(IplImage* tongueCoat, IplImage* tongueNature,
           CvScalar tongueCoatFeature, CvScalar tongueNatureFeature,
           CvRect coatRect) {
  IplImage* tongueCoat_Lab = NULL;
  IplImage* tongueCoat_L = NULL;
  IplImage* tongueCoat_A = NULL;
  IplImage* tongueCoat_B = NULL;
  IplImage* tongueNature_Lab = NULL;
  IplImage* tongueNature_L = NULL;
  IplImage* tongueNature_A = NULL;
  IplImage* tongueNature_B = NULL;

  tongueCoat_Lab = cvCreateImage(cvGetSize(tongueCoat), 8, 3);
  tongueNature_Lab = cvCreateImage(cvGetSize(tongueNature), 8, 3);
  tongueCoat_L = cvCreateImage(cvGetSize(tongueCoat), 8, 1);
  tongueCoat_A = cvCreateImage(cvGetSize(tongueCoat), 8, 1);
  tongueCoat_B = cvCreateImage(cvGetSize(tongueCoat), 8, 1);
  tongueNature_L = cvCreateImage(cvGetSize(tongueNature), 8, 1);
  tongueNature_A = cvCreateImage(cvGetSize(tongueNature), 8, 1);
  tongueNature_B = cvCreateImage(cvGetSize(tongueNature), 8, 1);

  cvCvtColor(tongueCoat, tongueCoat_Lab, CV_BGR2Lab);
  cvCvtColor(tongueNature, tongueNature_Lab, CV_BGR2Lab);
  cvSplit(tongueCoat_Lab, tongueCoat_L, tongueCoat_A, tongueCoat_B, NULL);
  cvSplit(tongueNature_Lab, tongueNature_L, tongueNature_A, tongueNature_B,
          NULL);

  for (int y = 0; y < tongueCoat_L->height; y++) {
    for (int x = 0; x < tongueCoat_L->width; x++) {
      if (cvGetReal2D(tongueCoat_L, y, x) == 255) {
        cvSetReal2D(tongueCoat_A, y, x, 255);
        cvSetReal2D(tongueCoat_B, y, x, 255);
      }
      if (cvGetReal2D(tongueNature_L, y, x) == 255) {
        cvSetReal2D(tongueNature_A, y, x, 255);
        cvSetReal2D(tongueNature_B, y, x, 255);
      }
    }
  }
  int nature_a = tongueNatureFeature.val[0];
  int nature_b = tongueNatureFeature.val[1];
  int coat_a = tongueCoatFeature.val[0];
  int coat_b = tongueCoatFeature.val[1];
  int distance_a = cvRound(abs(nature_a - coat_a) / 2);
  int distance_b = cvRound(abs(nature_b - coat_b) / 2);

  int coatNum = 0;
  int dcrement = 0;
  float rate = 0.0;
  for (int y = coatRect.y; y < coatRect.y + coatRect.height; y++) {
    for (int x = coatRect.x; x < coatRect.x + coatRect.width; x++) {
      int intensity = 0;
      int coat_a = 0;
      int coat_b = 0;
      coat_a = cvGetReal2D(tongueCoat_A, y, x);
      coat_b = cvGetReal2D(tongueCoat_B, y, x);
      intensity = cvRound(cvGetReal2D(tongueCoat_L, y, x));
      if (intensity != 255) {
        coatNum++;
      }
      if (abs(coat_b - nature_b) <= distance_b ||
          abs(coat_a - nature_a) <= distance_a) {
        if (intensity != 255) {
          dcrement++;
          cvSet2D(tongueCoat, y, x, cvScalar(255, 255, 255));
        }
      }
    }
  }
  rate = ((float)dcrement) / ((float)coatNum);

  cvReleaseImage(&tongueCoat_Lab);
  cvReleaseImage(&tongueCoat_L);
  cvReleaseImage(&tongueCoat_A);
  cvReleaseImage(&tongueCoat_B);
  cvReleaseImage(&tongueNature_Lab);
  cvReleaseImage(&tongueNature_L);
  cvReleaseImage(&tongueNature_A);
  cvReleaseImage(&tongueNature_B);

  return rate;
}
vector<int> canshu(IplImage* in) {
  for (int y = 0; y < in->height; y++) {
    for (int x = 0; x < in->width; x++) {
      CvScalar value = cvGet2D(in, y, x);
      if (value.val[0] == 0 && value.val[1] == 0 && value.val[2] == 0) {
        cvSet2D(in, y, x, cvScalar(255, 255, 255));
      }
    }
  }

  int L1, a1, b1, L2, a2, b2, L3, a3, b3;
  vector<int> returnres;
  IplImage* gray = cvCreateImage(cvGetSize(in), 8, 1);
  IplImage* copy = cvCreateImage(cvGetSize(in), 8, 3);
  IplImage* bottom = cvCreateImage(cvGetSize(in), 8, 3);
  cvCvtColor(in, gray, CV_RGB2GRAY);
  for (int i = 0; i < gray->height; i++) {
    for (int j = 0; j < gray->width; j++) {
      if (cvGetReal2D(gray, i, j) < 5)
        cvSet2D(in, i, j, cvScalar(255, 255, 255));
    }
  }
  cvCopy(in, copy);
  cvCopy(in, bottom);
  cvCvtColor(in, gray, CV_RGB2GRAY);
  IplImage* threshold = cvCreateImage(cvGetSize(in), 8, 1);
  cvThreshold(gray, threshold, 100, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(threshold, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  CvRect rect0 = cvRect(0, 0, 0, 0);
  int area0 = 0;
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);

    if (area > area0) {
      area0 = area;
      rect0 = rect;
    }
  }

  for (int i = 0; i < bottom->height; i++) {
    for (int j = 0; j < bottom->width; j++) {
      if (i > (rect0.y) && i < (rect0.height * 2 / 3 + rect0.y) &&
          j > (rect0.x + rect0.width / 6) &&
          j < (rect0.x + rect0.width * 5 / 6)) {
        cvSet2D(bottom, i, j, cvScalar(255, 255, 255));
      }
    }
  }
  IplImage* LAB2 = cvCreateImage(cvGetSize(copy), 8, 3);
  IplImage* LL2 = cvCreateImage(cvGetSize(copy), 8, 1);
  IplImage* aa2 = cvCreateImage(cvGetSize(copy), 8, 1);
  IplImage* bb2 = cvCreateImage(cvGetSize(copy), 8, 1);
  cvCvtColor(bottom, LAB2, CV_RGB2Lab);
  cvSplit(LAB2, LL2, aa2, bb2, NULL);
  for (int i = 0; i < LL2->height; i++) {
    for (int j = 0; j < LL2->width; j++) {
      if (cvGetReal2D(LL2, i, j) == 255) {
        cvSetReal2D(aa2, i, j, 255);
        cvSetReal2D(bb2, i, j, 255);
      }
    }
  }

  a2 = colorhist5(aa2);
  b2 = colorhist5(bb2);
  L2 = colorhist5(LL2);

  cvSetImageROI(copy, cvRect(rect0.x + rect0.width / 6, rect0.y,
                             rect0.width * 2 / 3, rect0.height * 2 / 3));

  IplImage* LAB = cvCreateImage(cvGetSize(copy), 8, 3);
  IplImage* LL = cvCreateImage(cvGetSize(copy), 8, 1);
  IplImage* aa = cvCreateImage(cvGetSize(copy), 8, 1);
  IplImage* bb = cvCreateImage(cvGetSize(copy), 8, 1);
  cvCvtColor(copy, LAB, CV_RGB2Lab);
  cvSplit(LAB, LL, aa, bb, NULL);
  for (int i = 0; i < LL->height; i++) {
    for (int j = 0; j < LL->width; j++) {
      if (cvGetReal2D(LL, i, j) == 255) {
        cvSetReal2D(aa, i, j, 255);
        cvSetReal2D(bb, i, j, 255);
      }
    }
  }

  L1 = colorhist5(LL);
  a1 = colorhist5(aa);
  b1 = colorhist5(bb);

  IplImage* shetai = cvCreateImage(cvGetSize(copy), 8, 3);
  cvCopy(copy, shetai);

  cvCvtColor(shetai, LAB, CV_RGB2Lab);
  cvSplit(LAB, LL, aa, bb, NULL);
  for (int i = 0; i < LL->height; i++) {
    for (int j = 0; j < LL->width; j++) {
      if (cvGetReal2D(LL, i, j) == 255) {
        cvSetReal2D(aa, i, j, 255);
        cvSetReal2D(bb, i, j, 255);
      }
    }
  }
  for (int i = 0; i < shetai->height; i++) {
    for (int j = 0; j < shetai->width; j++) {
      a3 = cvGetReal2D(aa, i, j);
      b3 = cvGetReal2D(bb, i, j);
      if (abs(a3 - a2) < 6) cvSet2D(shetai, i, j, cvScalar(255, 255, 255));
    }
  }
  cvCvtColor(shetai, LAB, CV_RGB2Lab);
  cvSplit(LAB, LL, aa, bb, NULL);
  int area = 0;
  for (int i = 0; i < LL->height; i++) {
    for (int j = 0; j < LL->width; j++) {
      if (cvGetReal2D(LL, i, j) == 255) {
        cvSetReal2D(aa, i, j, 255);
        cvSetReal2D(bb, i, j, 255);
      } else {
        ++area;
      }
    }
  }

  L3 = colorhist5(LL);
  a3 = colorhist5(aa);
  b3 = colorhist5(bb);
  cvResetImageROI(copy);
  returnres.push_back(L2);
  returnres.push_back(a2);
  returnres.push_back(b2);
  returnres.push_back(L3);
  returnres.push_back(a3);
  returnres.push_back(b3);

  return returnres;
}
int colorhist5(IplImage* src) {
  IplImage* gray_plane = cvCreateImage(cvGetSize(src), 8, 1);
  if (src->nChannels == 3) {
    cvCvtColor(src, gray_plane, CV_BGR2GRAY);
  } else {
    cvCopy(src, gray_plane);
  }

  int hist_size = 256;
  int hist_height = 256;
  float range[] = {0, 255};
  float* ranges[] = {range};
  CvHistogram* gray_hist =
      cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);

  cvCalcHist(&gray_plane, gray_hist, 0, 0);

  cvNormalizeHist(gray_hist, 1.0);
  int scale = 2;
  IplImage* hist_image =
      cvCreateImage(cvSize(hist_size * scale, hist_height), 8, 3);
  cvZero(hist_image);
  float max_value = 0;
  cvGetMinMaxHistValue(gray_hist, 0, &max_value, 0, 0);
  int max;

  int intensity = 0;
  int temp = 0;
  for (int i = 0; i < hist_size; i++) {
    float bin_val = cvQueryHistValue_1D(gray_hist, i);
    intensity = cvRound(bin_val * hist_height / max_value);
    if (intensity > temp && i > 10 && i < 240) {
      max = i;
      temp = intensity;
    }
    cvRectangle(hist_image, cvPoint(i * scale, hist_height - 1),
                cvPoint((i + 1) * scale - 1, hist_height - intensity),
                CV_RGB(255, 255, 255));
  }

  return max;
}
CvScalar tsbd(IplImage* in, int& substanceResponse) {
  vector<int> feature;
  feature = canshu(in);

  int t1 = feature[0];
  int t2 = feature[1];
  int t3 = feature[2];
  int t4 = feature[3];
  int t5 = feature[4];
  int t6 = feature[5];

  CvScalar tongueNatureColorFeature = cvScalar(t1, t2, t3);

  string shexiang_color = "";

  if (shexiang_color == "" && t1 <= 105) {
    shexiang_color = "�వ��";
    substanceResponse = 0;
  }
  if (shexiang_color == "" && t2 <= 136) {
    shexiang_color = "�൭��";
    substanceResponse = 1;
  }
  if (shexiang_color == "" && t2 <= 140) {
    shexiang_color = "�൭��";
    substanceResponse = 2;
  }
  if (shexiang_color == "" && t2 <= 148) {
    shexiang_color = "���";
    substanceResponse = 3;
  }
  if (shexiang_color == "" && t2 > 148) {
    substanceResponse = 4;
    shexiang_color = "�����";
  }

  return tongueNatureColorFeature;
}
void tctd(IplImage* tongueCoatImage1, IplImage* tongueNatureImage1) {

  tongueCoatImage = cvCloneImage(tongueCoatImage1);
  tongueNatureImage = cvCloneImage(tongueNatureImage1);

  feature[0] = 0.0;
  feature[1] = 0.0;
}
int prbh() {
  exfe();

  int response = 0;
  Mat testDataMat(1, 2, CV_32FC1, feature);

  //CvSVM svm = CvSVM();
//  CvSVM svm;
//  svm.load(bohouClassifierPathName);
//  response = (int)svm.predict(testDataMat);
  Mat responses;
  Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(bohouClassifierPathName); //读取模型
  svm->predict(testDataMat, responses);
  responses.convertTo(responses,CV_32S);
  response = responses.at<int>(0,0);
  return response;
}
void tcns(IplImage* tongueImage1, IplImage* tongueMask1) {
  tongueImage = cvCloneImage(tongueImage1);
  tongueMask = cvCloneImage(tongueMask1);

  tongueCoat = NULL;
  tongueNature = NULL;
}
void pffl() {
  IplImage* tongueROI = tbre(tongueImage, tongueMask);

  IplImage* imageTongue_lab = cvCreateImage(cvGetSize(tongueROI), 8, 3);
  cvCvtColor(tongueROI, imageTongue_lab, CV_BGR2Lab);

  IplImage* imageClusterResult = NULL;
  imageClusterResult = cvCreateImage(cvGetSize(tongueROI), 8, 3);
  cicbm(imageTongue_lab, imageClusterResult, 3);

  tongueCoat = cvCloneImage(tongueROI);
  tongueNature = cvCloneImage(tongueROI);
  tose(tongueROI, imageClusterResult, tongueCoat, tongueNature);

  cvReleaseImage(&tongueROI);
  cvReleaseImage(&imageTongue_lab);
  cvReleaseImage(&imageClusterResult);
}
IplImage* ddccyy(IplImage* in, double lowThresh, double highThresh,
                 double aperture) {
  if (in->nChannels != 1) return (0);
  IplImage* out = cvCreateImage(cvGetSize(in), in->depth, 1);
  cvCanny(in, out, lowThresh, highThresh, aperture);
  return (out);
};
IplImage* tbre(IplImage* in, IplImage* mask) {
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(mask, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  int max_area = 0;
  CvRect tongueRect = cvRect(0, 0, 0, 0);
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);
    if (max_area < area) {
      max_area = area;
      tongueRect = rect;
    }
  }

  tongueRect.y -= 5;
  tongueRect.height += 5;
  cvSetImageROI(in, tongueRect);
  IplImage* tongueROI = cvCreateImage(cvGetSize(in), 8, 3);
  cvCopy(in, tongueROI);
  cvResetImageROI(in);

  cvReleaseMemStorage(&pStorage);
  return tongueROI;
}
void cicbm(const IplImage* imageSrc, IplImage* imageResult, int cluster_count) {
  CvMat* samples = NULL;
  CvMat* labels = NULL;
  samples = cvCreateMat(imageSrc->width * imageSrc->height, 1, CV_32FC2);
  labels = cvCreateMat(imageSrc->width * imageSrc->height, 1, CV_32SC1);
  int k = 0;
  for (int y = 0; y < imageSrc->height; y++) {
    for (int x = 0; x < imageSrc->width; x++) {
      CvScalar srcPixel;
      CvScalar samplesPixel = cvScalar(0.0);
      srcPixel = cvGet2D(imageSrc, y, x);
      samplesPixel.val[0] = srcPixel.val[1];
      samplesPixel.val[1] = srcPixel.val[2];
      cvSet2D(samples, k++, 0, samplesPixel);
    }
  }
  cvKMeans2(samples, cluster_count, labels,
            cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1.0));

  CvScalar color_label[3];
  color_label[0] = CV_RGB(255, 0, 0);
  color_label[1] = CV_RGB(0, 255, 0);
  color_label[2] = CV_RGB(0, 0, 255);
  k = 0;
  for (int y = 0; y < imageSrc->height; y++) {
    for (int x = 0; x < imageSrc->width; x++) {
      int cluster_id;
      cluster_id = cvGetReal2D(labels, k++, 0);
      cvSet2D(imageResult, y, x, color_label[cluster_id]);
    }
  }

  cvReleaseMat(&samples);
  cvReleaseMat(&labels);
}
void tose(IplImage* imageSrc, IplImage* imageResult, IplImage* tongueCoat,
          IplImage* tongueNature) {
  const int blue = 0, green = 1, red = 2;
  int backgroundColor = 0, tongueCoatColor = 0, tongueNatureColor = 0;

  if (cvGet2D(imageResult, 2, 2).val[0] > 200) {
    backgroundColor = blue;
  } else if (cvGet2D(imageResult, 2, 2).val[1] > 200) {
    backgroundColor = green;
  } else if (cvGet2D(imageResult, 2, 2).val[2] > 200) {
    backgroundColor = red;
  }

  int k = imageResult->width / 2;
  for (int y = 0; y < imageResult->height - 3; y++) {
    if (backgroundColor != 0 && (cvGet2D(imageResult, y, k).val[0] == 255) &&
        (cvGet2D(imageResult, y + 1, k).val[0] == 255) &&
        (cvGet2D(imageResult, y + 2, k).val[0] == 255)) {
      tongueCoatColor = blue;
      break;
    }
    if (backgroundColor != 1 && (cvGet2D(imageResult, y, k).val[1] == 255) &&
        (cvGet2D(imageResult, y + 1, k).val[1] == 255) &&
        (cvGet2D(imageResult, y + 2, k).val[1] == 255)) {
      tongueCoatColor = green;
      break;
    }
    if (backgroundColor != 2 && (cvGet2D(imageResult, y, k).val[2] == 255) &&
        (cvGet2D(imageResult, y + 1, k).val[2] == 255) &&
        (cvGet2D(imageResult, y + 2, k).val[2] == 255)) {
      tongueCoatColor = red;
      break;
    }
  }
  if (blue != backgroundColor && blue != tongueCoatColor) {
    tongueNatureColor = blue;
  }
  if (green != backgroundColor && green != tongueCoatColor) {
    tongueNatureColor = green;
  }
  if (red != backgroundColor && red != tongueCoatColor) {
    tongueNatureColor = red;
  }
  for (int y = 0; y < imageResult->height; y++) {
    for (int x = 0; x < imageResult->width; x++) {
      if (cvGet2D(imageResult, y, x).val[tongueCoatColor] != 255) {
        cvSet2D(tongueCoat, y, x, CV_RGB(255, 255, 255));
      }
      if (cvGet2D(imageResult, y, x).val[tongueNatureColor] != 255) {
        cvSet2D(tongueNature, y, x, CV_RGB(255, 255, 255));
      }
    }
  }
}
IplImage* gtci() { return tongueCoat; }
IplImage* gtni() { return tongueNature; }
int LSF(vector<CvPoint2D64f> vec, int index_power) {
  if (vec.size() > 1000) {
    return 0;
  }
  double x[1000] = {0}, y[1000] = {0}, max = 0, max2 = 0;
  double sumX[1000] = {0}, sumY[1000] = {0};
  int pre_label = -1;
  int i, j, n, index, ii = 1, res;
  CvPoint center = cvPoint(0, 0);
  int r = 1;
  for (vector<CvPoint2D64f>::iterator itr = vec.begin(); itr != vec.end();
       ++itr) {
    x[ii] = (*itr).x;
    y[ii] = (*itr).y;
    if (x[ii] > max) max = x[ii];
    if (y[ii] > max) max = y[ii];
    ii++;
  }
  n = ii - 1;
  sumX[1] = sumY[1] = 0;
  for (i = 1; i <= n; i++) {
    sumX[1] += x[i];
    sumY[1] += y[i];
    x[i] /= max;
    y[i] /= max;
  }
  index = index_power;
  i = n;
  sumX[0] = i;
  for (i = 2; i <= 2 * index; i++) {
    for (j = 1; j <= n; j++) {
      sumX[i] += power(x[j], i);
    }
  }
  for (i = 2; i <= index + 1; i++) {
    for (j = 1; j <= n; j++) {
      sumY[i] += power(x[j], i - 1) * y[j];
    }
  }
  double a[10][11], a0[10][11];
  double l[10];

  for (int i = 0; i < 10; i++) {
    l[i] = 0;
    for (int j = 0; j < 11; j++) {
      a[i][j] = 0;
      a0[i][j] = 0;
    }
  }
  int N, k;
  double sum;
  N = index + 1;
  for (i = 0; i <= index; i++) {
    k = 1;
    for (j = i; j <= index + i; j++) a[i + 1][k++] = sumX[j];
    a[i + 1][k++] = sumY[i + 1];
  }

  for (i = 1; i <= N; i++) {
    for (j = 1; j <= N + 1; j++) {
      a0[i][j] = a[i][j];
    }
  }

  k = 1;
  do {
    for (i = k + 1; i <= N; i++) {
      l[i] = a0[i][k] / a0[k][k];
      for (j = 1; j <= N + 1; j++) {
        a[i][j] = a0[i][j] - l[i] * a0[k][j];
      }
    }
    if (k == N) break;
    k++;
    for (j = 1; j <= N; j++) {
      for (i = 1; i <= N + 1; i++) a0[j][i] = a[j][i];
    }
  } while (true);
  l[N] = a[N][N + 1] / a[N][N];
  for (k = N - 1; k >= 1; k--) {
    sum = 0;
    for (j = k + 1; j <= N; j++) sum += a[k][j] * l[j];
    l[k] = (a[k][N + 1] - sum) / a[k][k];
  }
  float data[4];
  for (int i = 2; i <= N; i++) {
    data[i - 2] = l[i];
  }

  pre_label = ttpt(data);

  return pre_label;
}
void tftd() {}
IplImage* fdrt(IplImage* image) {
  if (image == NULL) {
    std::cout << "read imagefailed" << std::endl;
  }
  CvMemStorage* storage = cvCreateMemStorage(0);
  CvSeq* contours = NULL;
  IplImage* dst = cvCloneImage(image);
  cvFindContours(dst, storage, &contours, sizeof(CvContour), CV_RETR_LIST,
                 CV_CHAIN_APPROX_NONE);
  CvRect maxRec = cvRect(0, 0, 0, 0);
  double maxArea = 0;

  for (; contours != NULL; contours = contours->h_next) {
    CvRect rect = cvBoundingRect(contours, 1);
    double tmparea = fabs(cvContourArea(contours));
    if (tmparea > maxArea) {
      maxArea = tmparea;
      maxRec = rect;
    }
  }
  cvSetImageROI(image, maxRec);
  IplImage* dst2 =
      cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
  cvCopy(image, dst2);
  cvResetImageROI(image);

  cvReleaseImage(&dst);
  cvReleaseMemStorage(&storage);

  return dst2;
}
int ttpt(float data[]) {
  int response;
  Mat testDataMat(1, 4, CV_32FC1, data);

//  CvSVM svm;
//  svm.load(panClassifierPathName);
//  response = (int)svm.predict(testDataMat);
  Mat responses;
  Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(panClassifierPathName); //读取模型
  svm->predict(testDataMat, responses);
  responses.convertTo(responses,CV_32S);
  response = responses.at<int>(0,0);
  return response;
}
float power(float a, int n) {
  float b = 1;
  for (int i = 0; i < n; i++) {
    b *= a;
  }
  return b;
}
int prdt(IplImage* tongueMask) {
  IplImage* tongueRoi = fdrt(tongueMask);

  IplImage* img_cny = ddccyy(tongueRoi, 10, 100, 3);
  cvSetImageROI(img_cny, cvRect(0, 0.3 * img_cny->height, img_cny->width,
                                0.7 * img_cny->height));
  IplImage* dst = cvCreateImage(cvGetSize(img_cny), 8, img_cny->nChannels);
  cvCopy(img_cny, dst);
  cvResetImageROI(img_cny);

  vector<CvPoint2D64f> vec;
  for (int y = 0; y < dst->height; y++) {
    for (int x = 0; x < dst->width; x++) {
      if (cvGetReal2D(dst, y, x) > 200) {
        vec.push_back(cvPoint2D64f(x, y));
      }
    }
  }

  int pre_label = -1;
  pre_label = LSF(vec, 4);

  cvReleaseImage(&tongueRoi);
  cvReleaseImage(&img_cny);
  cvReleaseImage(&dst);

  return pre_label;
}
void ggff(IplImage* I1, IplImage* p1, int r1, double eps1) {
  I = cvCreateImage(cvGetSize(I1), IPL_DEPTH_32F, 1);
  nmlz(I1, I);
  p = cvCreateImage(cvGetSize(p1), IPL_DEPTH_32F, 1);
  nmlz(p1, p);
  rlvbo = r1;
  eps = eps1;
  N = (2 * rlvbo + 1) * (2 * rlvbo + 1);
  q = cvCreateImage(cvGetSize(I1), IPL_DEPTH_32F, 1);
}
void caqq() {
  IplImage* mean_I;
  IplImage* mean_p;
  IplImage* mean_Ip;
  IplImage* cov_Ip;
  IplImage* mean_II;
  IplImage* var_I;
  IplImage* a;
  IplImage* b;
  IplImage* mean_a;
  IplImage* mean_b;

  mean_I = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  mean_p = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  mean_Ip = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cov_Ip = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  mean_II = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  var_I = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  a = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  b = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  mean_a = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  mean_b = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);

  cvSmooth(I, mean_I, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);

  cvSmooth(p, mean_p, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);

  cvMul(I, p, mean_Ip);
  cvSmooth(mean_Ip, mean_Ip, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);

  IplImage* Mul_meanI_mean_p = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cvMul(mean_I, mean_p, Mul_meanI_mean_p);
  cvAbsDiff(mean_Ip, Mul_meanI_mean_p, cov_Ip);
  cvReleaseImage(&Mul_meanI_mean_p);

  IplImage* II = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cvMul(I, I, II);
  cvSmooth(II, mean_II, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);
  cvReleaseImage(&II);

  IplImage* Mul_meanI_mean_I = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cvMul(mean_I, mean_I, Mul_meanI_mean_I);
  cvAbsDiff(mean_II, Mul_meanI_mean_I, var_I);
  cvReleaseImage(&Mul_meanI_mean_I);
  cvAddS(var_I, cvScalarAll(eps), var_I);
  cvDiv(cov_Ip, var_I, a);

  IplImage* Mul_a_mean_I = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cvMul(a, mean_I, Mul_a_mean_I);
  cvAbsDiff(mean_p, Mul_a_mean_I, b);
  cvReleaseImage(&Mul_a_mean_I);

  cvSmooth(a, mean_a, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);

  cvSmooth(b, mean_b, CV_BLUR, 2 * rlvbo + 1, 2 * rlvbo + 1);

  IplImage* Mul_mean_a_I = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
  cvMul(mean_a, I, Mul_mean_a_I);
  cvAdd(Mul_mean_a_I, mean_b, q);
  cvReleaseImage(&Mul_mean_a_I);

  cvReleaseImage(&mean_I);
  cvReleaseImage(&mean_p);
  cvReleaseImage(&mean_Ip);
  cvReleaseImage(&cov_Ip);
  cvReleaseImage(&mean_II);
  cvReleaseImage(&var_I);
  cvReleaseImage(&a);
  cvReleaseImage(&b);
  cvReleaseImage(&mean_a);
  cvReleaseImage(&mean_b);
}

IplImage* ggqq() {
  caqq();
  return q;
}
void nmlz(IplImage* src, IplImage* dst) {
  for (int y = 0; y < src->height; y++) {
    for (int x = 0; x < src->width; x++) {
      double pixel = cvGetReal2D(src, y, x) / 255;
      cvSetReal2D(dst, y, x, pixel);
    }
  }
}
void rghcu(IplImage* src) {
  inputImagecucao = cvCreateImage(cvGetSize(src), src->depth, 1);
  if (src->nChannels > 1) {
    cvCvtColor(src, inputImagecucao, CV_BGR2GRAY);
  } else {
    inputImagecucao = cvCloneImage(src);
  }

  rcucao = 5;

  win_mat =
      cvCreateImage(cvSize(2 * rcucao + 1, 2 * rcucao + 1), IPL_DEPTH_8U, 1);

  image_sdv = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  cvZero(image_sdv);

  image_roughness = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  cvZero(image_roughness);
}
void carh() {
  for (int y = rcucao; y < inputImagecucao->height - rcucao; y++) {
    for (int x = rcucao; x < inputImagecucao->width - rcucao; x++) {
      cnbp(x, y);

      CvScalar mean = cvScalar(0);
      CvScalar sdv = cvScalar(0);
      cvAvgSdv(win_mat, &mean, &sdv);

      cvSetReal2D(image_sdv, y, x, sdv.val[0]);
    }
  }

  cvThreshold(image_sdv, image_roughness, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);
}
void cnbp(int x0, int y0) {
  CvRect rect =
      cvRect(x0 - rcucao, y0 - rcucao, 2 * rcucao + 1, 2 * rcucao + 1);
  cvSetImageROI(inputImagecucao, rect);
  cvCopy(inputImagecucao, win_mat);
  cvResetImageROI(inputImagecucao);
}
IplImage* girh() { return image_roughness; }
void ddkk(IplImage* src) {
  inputImageandu = cvCreateImage(cvGetSize(src), src->depth, 1);
  if (src->nChannels > 1) {
    cvCvtColor(src, inputImageandu, CV_BGR2GRAY);
  } else {
    inputImageandu = cvCloneImage(src);
  }

  rdu = 5;

  darkImageBi = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
}
void cadk() {
  IplImage* darkImage;
  darkImage = cvCreateImage(cvGetSize(inputImageandu), IPL_DEPTH_8U, 1);
  IplConvKernel* patch;
  patch = cvCreateStructuringElementEx(2 * rdu + 1, 2 * rdu + 1, rdu, rdu,
                                       CV_SHAPE_RECT);

  cvErode(inputImageandu, darkImage, patch, 1);

  cvThreshold(darkImage, darkImageBi, 0, 255,
              CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

  cvReleaseImage(&darkImage);
  cvReleaseStructuringElement(&patch);
}
IplImage* gdib() { return darkImageBi; }
void wwff(IplImage* giudedFilterImage, IplImage* imageRoughness) {
  inputImageshuiliu =
      cvCreateImage(cvGetSize(giudedFilterImage), giudedFilterImage->depth, 1);
  inputImageshuiliu = cvCloneImage(giudedFilterImage);

  inputRoughness =
      cvCreateImage(cvGetSize(imageRoughness), imageRoughness->depth, 1);
  inputRoughness = cvCloneImage(imageRoughness);

  L = cvCreateImage(cvGetSize(giudedFilterImage), 8, 1);
  cvZero(L);
}
void psfw() {
  catt();
  calll();
}
void calll() {
  for (int y0 = 0; y0 < inputImageshuiliu->height; y0++) {
    for (int x0 = 0; x0 < inputImageshuiliu->width; x0++) {
      if (cvRound(cvGetReal2D(inputRoughness, y0, x0)) > 200) {
        pwff(x0, y0);
      }
    }
  }
  cagg();
  for (int y = 0; y < L->height; y++) {
    for (int x = 0; x < L->width; x++) {
      double value = cvGetReal2D(L, y, x);
      if (value > g) {
      } else {
        cvSetReal2D(L, y, x, 0);
      }
    }
  }

  cvThreshold(L, L, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}
void cagg() {
  float data[256] = {0};
  int total = 0;
  for (int y = 0; y < L->height; y++) {
    for (int x = 0; x < L->width; x++) {
      if (cvGetReal2D(L, y, x) > 0) {
        int intensity = cvRound(cvGetReal2D(L, y, x));
        data[intensity]++;
        total++;
      }
    }
  }
  float sum_dark = 0;
  double rate_dark = 0.2;
  for (int i = 0; i < 256; i++) {
    sum_dark += data[i];
    if ((sum_dark) / ((double)total) > rate_dark) {
      g = i;
      break;
    }
  }
}
void catt() {
  IplImage* image_difference = cvCreateImage(cvGetSize(inputRoughness), 8, 1);
  cvZero(image_difference);
  float data[256] = {0};
  int total = 0;

  for (int y = 0; y < inputImageshuiliu->height; y++) {
    for (int x = 0; x < inputImageshuiliu->width; x++) {
      if (cvRound(cvGetReal2D(inputRoughness, y, x)) > 200) {
        int currentPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x));
        int d = 1;
        int maxDifference = -300;
        int maxabsDifference = 0;

        if ((x - d >= 0) && (y >= 0) && (x - d < inputImageshuiliu->width) &&
            (y < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x - d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x - d >= 0) && (y - d >= 0) &&
            (x - d < inputImageshuiliu->width) &&
            (y - d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x - d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x >= 0) && (y - d >= 0) && (x < inputImageshuiliu->width) &&
            (y - d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x + d >= 0) && (y - d >= 0) &&
            (x + d < inputImageshuiliu->width) &&
            (y - d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x + d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x + d >= 0) && (y >= 0) && (x + d < inputImageshuiliu->width) &&
            (y < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x + d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x + d >= 0) && (y + d >= 0) &&
            (x + d < inputImageshuiliu->width) &&
            (y + d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y + d, x + d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x >= 0) && (y + d >= 0) && (x < inputImageshuiliu->width) &&
            (y + d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y + d, x));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }

        if ((x - d >= 0) && (y + d >= 0) &&
            (x - d < inputImageshuiliu->width) &&
            (y + d < inputImageshuiliu->height)) {
          int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x - d));
          int difference = currentPixel - nextPixel;
          if (difference > maxDifference) {
            maxDifference = difference;
          }
          if (maxabsDifference < abs(difference)) {
            maxabsDifference = abs(difference);
          }
        }
        cvSetReal2D(image_difference, y, x, abs(maxDifference));
        if (maxabsDifference != 0) {
          data[maxabsDifference]++;
          total++;
        }
      }
    }
  }

  float sum = 0;
  double rate = 0.15;
  for (int i = 0; i < 256; i++) {
    sum += data[i];
    if ((sum) / ((double)total) > rate) {
      T = -(i - 1);
      if (T >= -1) {
        T = -3;
      }
      if (T <= -9) {
        T = -9;
      }
      break;
    }
  }
}
void pwff(int x, int y) {
  IplImage* flag =
      cvCreateImage(cvSize(inputImageshuiliu->width, inputImageshuiliu->height),
                    IPL_DEPTH_8U, 1);
  cvZero(flag);

  while (true) {
    int currentPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x));
    cvSetReal2D(flag, y, x, 1);

    int difference[8] = {-1000, -1000, -1000, -1000,
                         -1000, -1000, -1000, -1000};
    CvPoint nextpoint[8];

    int d = 1;
    if ((x - d >= 0) && (y >= 0) && (x - d < inputImageshuiliu->width) &&
        (y < inputImageshuiliu->height)) {
      nextpoint[0].x = x - d;
      nextpoint[0].y = y;
      if (!cvGetReal2D(flag, y, x - d)) {
        cvSetReal2D(flag, y, x - d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x - d));
        difference[0] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x - d >= 0) && (y - d >= 0) && (x - d < inputImageshuiliu->width) &&
        (y - d < inputImageshuiliu->height)) {
      nextpoint[1].x = x - d;
      nextpoint[1].y = y - d;
      if (!cvGetReal2D(flag, y - d, x - d)) {
        cvSetReal2D(flag, y - d, x - d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x - d));
        difference[1] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x >= 0) && (y - d >= 0) && (x < inputImageshuiliu->width) &&
        (y - d < inputImageshuiliu->height)) {
      nextpoint[2].x = x;
      nextpoint[2].y = y - d;
      if (!cvGetReal2D(flag, y - d, x)) {
        cvSetReal2D(flag, y - d, x, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x));
        difference[2] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x + d >= 0) && (y - d >= 0) && (x + d < inputImageshuiliu->width) &&
        (y - d < inputImageshuiliu->height)) {
      nextpoint[3].x = x + d;
      nextpoint[3].y = y - d;
      if (!cvGetReal2D(flag, y - d, x + d)) {
        cvSetReal2D(flag, y - d, x + d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y - d, x + d));
        difference[3] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x + d >= 0) && (y >= 0) && (x + d < inputImageshuiliu->width) &&
        (y < inputImageshuiliu->height)) {
      nextpoint[4].x = x + d;
      nextpoint[4].y = y;
      if (!cvGetReal2D(flag, y, x + d)) {
        cvSetReal2D(flag, y, x + d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y, x + d));
        difference[4] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x + d >= 0) && (y + d >= 0) && (x + d < inputImageshuiliu->width) &&
        (y + d < inputImageshuiliu->height)) {
      nextpoint[5].x = x + d;
      nextpoint[5].y = y + d;
      if (!cvGetReal2D(flag, y + d, x + d)) {
        cvSetReal2D(flag, y + d, x + d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y + d, x + d));
        difference[5] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x >= 0) && (y + d >= 0) && (x < inputImageshuiliu->width) &&
        (y + d < inputImageshuiliu->height)) {
      nextpoint[6].x = x;
      nextpoint[6].y = y + d;
      if (!cvGetReal2D(flag, y + d, x)) {
        cvSetReal2D(flag, y + d, x, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y + d, x));
        difference[6] = currentPixel - nextPixel;
      } else {
      }
    }

    if ((x - d >= 0) && (y + d >= 0) && (x - d < inputImageshuiliu->width) &&
        (y + d < inputImageshuiliu->height)) {
      nextpoint[7].x = x - d;
      nextpoint[7].y = y + d;
      if (!cvGetReal2D(flag, y + d, x - d)) {
        cvSetReal2D(flag, y + d, x - d, 1);
        int nextPixel = cvRound(cvGetReal2D(inputImageshuiliu, y + d, x - d));
        difference[7] = currentPixel - nextPixel;
      } else {
      }
    }
    int maxDifference = -300;
    int maxAbsDifference = 0;
    CvPoint flowDirection;
    for (int i = 0; i < 8; i++) {
      if (maxDifference < difference[i]) {
        maxDifference = difference[i];
        flowDirection.x = nextpoint[i].x;
        flowDirection.y = nextpoint[i].y;
      }
      if (maxAbsDifference < abs(difference[i]) && abs(difference[i]) < 255) {
        maxAbsDifference = abs(difference[i]);
      }
    }
    int flagy = y - d;
    if (flagy < 0) {
      flagy = 0;
    }
    int flagx = x - d;
    if (flagx < 0) {
      flagx = 0;
    }

    for (; flagy - y + d < 2 * d + 1 && flagy < flag->height; flagy++) {
      for (; flagx - x + d < 2 * d + 1 && flagx < flag->width; flagx++) {
        cvSetReal2D(flag, flagy, flagx, 1);
      }
    }

    if ((x > 0) && (y > 0) && (x < inputImageshuiliu->width - 1) &&
        (y < inputImageshuiliu->height - 1) && (maxDifference < T)) {
      cvSetReal2D(L, y, x, cvGetReal2D(L, y, x) + maxAbsDifference);
      cvSetReal2D(inputImageshuiliu, y, x,
                  cvGetReal2D(inputImageshuiliu, y, x) + maxAbsDifference);

      break;
    } else if ((x == 0) || (y == 0) || (x == inputImageshuiliu->width - 1) ||
               (y == inputImageshuiliu->height - 1)) {
      break;
    } else {
      x = flowDirection.x;
      y = flowDirection.y;
    }
  }

  cvReleaseImage(&flag);
}
IplImage* ggll() { return L; }
int tcdd(IplImage* inputImage) {
  IplImage* imageNormal =
      cvCreateImage(cvSize(100, 100), inputImage->depth, inputImage->nChannels);
  cvResize(inputImage, imageNormal);

  IplImage* grayImage = cvCreateImage(cvGetSize(imageNormal), 8, 1);
  cvCvtColor(imageNormal, grayImage, CV_BGR2GRAY);

  ggff(grayImage, grayImage, 5, 0.001);
  IplImage* q = ggqq();

  IplImage* qq = cvCreateImage(cvGetSize(q), 8, 1);
  cvConvertScaleAbs(q, qq, 255, 0.0);

  rghcu(qq);
  carh();
  IplImage* imageRoughness = girh();

  ddkk(qq);
  cadk();
  IplImage* DarkImageBI = gdib();
  wwff(qq, imageRoughness);
  psfw();
  IplImage* image_L = ggll();
  for (int y = 0; y < image_L->height; y++) {
    for (int x = 0; x < image_L->width; x++) {
      if (cvGetReal2D(DarkImageBI, y, x) < 100) {
        cvSetReal2D(image_L, y, x, 0);
      }
    }
  }
  cvSmooth(image_L, image_L, CV_MEDIAN);
  cvMorphologyEx(image_L, image_L, NULL, NULL, CV_MOP_CLOSE);
  int areaMin = 30;
  int lenMin = 20;
  RemoveNoise removeNoise;
  int nCracks = removeNoise.RemoveCrackImageNoise(image_L, areaMin, lenMin);

  cvReleaseImage(&imageNormal);
  cvReleaseImage(&grayImage);
  cvReleaseImage(&qq);

  return nCracks;
}
void tdi(IplImage* imageSrc1) {

  imageSrcS = cvCloneImage(imageSrc1);
  if (storageS) {
      cvReleaseMemStorage(&storageS);
      storageS = 0;
  }
  cascadeS = 0;
}
bool dddss() {
  bool isTongue = false;
  cascadeS = (CvHaarClassifierCascade*)cvLoad(cascade_nameS, 0, 0, 0);
  storageS = cvCreateMemStorage(0);

  double scale = 1.0;
  IplImage* gray =
      cvCreateImage(cvSize(imageSrcS->width, imageSrcS->height), 8, 1);
  IplImage* small_img =
      cvCreateImage(cvSize(cvRound(imageSrcS->width / scale),
                           cvRound(imageSrcS->height / scale)),
                    8, 1);

  cvCvtColor(imageSrcS, gray, CV_BGR2GRAY);
  cvResize(gray, small_img, CV_INTER_LINEAR);
  cvClearMemStorage(storageS);

  CvRect *r, maxR;
  int rSize, maxSize = 0;
  int cnt = 0;
  if (cascadeS) {
    CvSeq* tongues = cvHaarDetectObjects(small_img, cascadeS, storageS, 1.1, 2,
                                         CV_HAAR_FIND_BIGGEST_OBJECT |
                                             CV_HAAR_DO_ROUGH_SEARCH |
                                             CV_HAAR_DO_CANNY_PRUNING,
                                         cvSize(10, 10));
    cnt = tongues->total;
    for (int i = 0; i < (tongues ? tongues->total : 0); i++) {
      r = (CvRect*)cvGetSeqElem(tongues, i);
      rSize = r->width * r->height;
      if (rSize > maxSize) {
        maxSize = rSize;
        maxR = *r;
        tongueRect = *r;
      }
    }
//    cout << "tongue detect cnt: " << cnt << endl;

  } else {
    cout << "Loading haarcascade_tongue.xml is wrong!!" << endl;
  }

  cvReleaseImage(&gray);
  cvReleaseImage(&small_img);
  if (storageS) {
      cvReleaseMemStorage(&storageS);
      storageS = 0;
  }
  if (cnt == 0) {
    maxR.x = 0;
    maxR.y = 0;
    maxR.width = 0;
    maxR.height = 0;
    isTongue = false;
  } else {
    isTongue = true;
    //int compareLength = small_img->width < small_img->height ? small_img->width : small_img->height;
    int compareLength = 200;
    int roiLength = tongueRect.width < tongueRect.height ? tongueRect.width : tongueRect.height;
    roiScale = compareLength * 1.0 / roiLength;
    tongueRect.x = tongueRect.x * scale;
    tongueRect.y = tongueRect.y * scale;
    tongueRect.height = tongueRect.height * scale;
    tongueRect.width = tongueRect.width * scale;
  }
  cvReleaseImage(&gray);
  cvReleaseImage(&small_img);
  return isTongue;
}

CvRect gtrt() { 
  //tongueRect.width += 10;
  //tongueRect.height += 10;
  return tongueRect; 
}

void ttssii(IplImage* in, int haarTonguedddsswitch /*=0*/) {
  switchHaarTongue = haarTonguedddsswitch;

  inputImageS = cvCloneImage(in);
  dyss();

  image_v = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  image_vBiPolar = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  image_vBiPolarcftt = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  image_vMask = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  image_tongue = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 3);

  isDetectTongue = true;
}
void dyss() {
  int size;
  size = (inputImageS->height > inputImageS->width) ? inputImageS->height
                                                    : inputImageS->width;
  int type = cvRound(((double)size) / 330.0);

  switch (type) {
    case 0:
      image_tongueROI = cvCloneImage(inputImageS);
      break;
    case 1:
      image_tongueROI = cvCloneImage(inputImageS);
      break;
    case 2:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 2, inputImageS->height / 2),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 3:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 3, inputImageS->height / 3),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 4:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 4, inputImageS->height / 4),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 5:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 5, inputImageS->height / 5),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 6:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 6, inputImageS->height / 6),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 7:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 7, inputImageS->height / 7),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 8:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 8, inputImageS->height / 8),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 9:
      image_tongueROI =
          cvCreateImage(cvSize(inputImageS->width / 9, inputImageS->height / 9),
                        inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 10:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 10, inputImageS->height / 10),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 11:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 11, inputImageS->height / 11),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 12:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 12, inputImageS->height / 12),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 13:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 13, inputImageS->height / 13),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 14:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 14, inputImageS->height / 14),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 15:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 15, inputImageS->height / 15),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 16:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 16, inputImageS->height / 16),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 17:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 17, inputImageS->height / 17),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 18:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 18, inputImageS->height / 18),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 19:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 19, inputImageS->height / 19),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    case 20:
      image_tongueROI = cvCreateImage(
          cvSize(inputImageS->width / 20, inputImageS->height / 20),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, image_tongueROI);
      break;
    default:
      image_tongueROI = cvCloneImage(inputImageS);
      break;
  }
}
bool ppff() {
  if (switchHaarTongue == 1) {
    int size;
    size = (inputImageS->height > inputImageS->width) ? inputImageS->height
                                                      : inputImageS->width;
    //int type = cvRound(((double)size) / 540.0);
    int type = 0;
    IplImage* imageScale;
    if (type == 0 || type == 1) {
      imageScale = cvCloneImage(inputImageS);
    } else {
      imageScale = cvCreateImage(
          cvSize(inputImageS->width / type, inputImageS->height / type),
          inputImageS->depth, inputImageS->nChannels);
      cvResize(inputImageS, imageScale);
    }

    tdi(imageScale);
    isDetectTongue = dddss();
    if (!isDetectTongue) {
      cvReleaseImage(&imageScale);
      return isDetectTongue;
    }
    cvReleaseImage(&imageScale);
  } else {
  }
  cgii();
  cgee();
  if (!ftcc()) {
    //return isDetectTongue;
  }
  ctmm();

  GMM *mGMM = bbmm(image_tongueROI);
  IplImage *tmpImage_Gauss = ccpp(image_tongueROI, mGMM);
  cvThreshold(tmpImage_Gauss,tmpImage_Gauss,0,255,CV_THRESH_BINARY);
  cvMorphologyEx(tmpImage_Gauss, image_vMask, NULL, NULL,CV_MOP_OPEN,4);
  RemoveNoise rn;
  rn.LessConnectedRegionRemove(image_vMask,
                               tmpImage_Gauss->height * tmpImage_Gauss->width / 8);

  sgtt();

  if (!ccmm()) {
//    return isDetectTongue;
  }

//  return isDetectTongue;
  return true;
}
void cgii() {
  CvRect tmpTRect = gtrt();
  cvSetImageROI(imageSrcS, tmpTRect);
  IplImage* tmpTongueROI = cvCreateImage(
      cvGetSize(imageSrcS), IPL_DEPTH_8U, 3);
  cvCopy(imageSrcS, tmpTongueROI);
  IplImage* tmpScaleTongueROI = cvCreateImage(
      cvSize(cvGetSize(tmpTongueROI).width*roiScale, cvGetSize(tmpTongueROI).height*roiScale), IPL_DEPTH_8U, 3);
  cvResize(tmpTongueROI, tmpScaleTongueROI);

  cvResetImageROI(imageSrcS);
  ttssii(tmpScaleTongueROI, 0);

  cvSmooth(tmpScaleTongueROI, tmpScaleTongueROI, CV_GAUSSIAN);
  IplImage* image_hsv = cvCreateImage(
      cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 3);
  IplImage* image_h = cvCreateImage(
      cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 1);
  IplImage* image_s = cvCreateImage(
      cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 1);
  cvCvtColor(tmpScaleTongueROI, image_hsv, CV_BGR2HSV);
  cvSplit(image_hsv, image_h, image_s, image_v, NULL);

  cvReleaseImage(&tmpTongueROI);
  cvReleaseImage(&tmpScaleTongueROI);
  cvReleaseImage(&image_hsv);
  cvReleaseImage(&image_h);
  cvReleaseImage(&image_s);
}
void cgee() {
  int M = 40;
  IplImage* image_vPolar = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  
  cvLogPolar(image_v, image_vPolar,
             cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,
             CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);

  IplImage* image_vPolarEdge = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  for (int y = 0; y < image_vPolar->height; y++) {
    for (int x = 0; x < image_vPolar->width; x++) {
      if ((x < 6) || (x > image_vPolar->width - 7)) {
        cvSetReal2D(image_vPolarEdge, y, x, 0);
      } else {
        int E = 0;
        for (int k = 1; k <= 6; k++) {
          E += cvRound(cvGet2D(image_vPolar, y, x + k).val[0]);
          E += cvRound(cvGet2D(image_vPolar, y, x - k).val[0]);
          E += -2 * cvGet2D(image_vPolar, y, x).val[0];
        }
        cvSetReal2D(image_vPolarEdge, y, x, E);
      }
    }
  }

  IplImage* image_vPolarEdgeInverse = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  cvLogPolar(image_vPolarEdge, image_vPolarEdgeInverse,
             cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,
             CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);

  IplImage* image_vPolarEdgeInverseOSTU = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  cvThreshold(image_vPolarEdgeInverse, image_vPolarEdgeInverseOSTU, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);  //�ɿ��ǲ��ô����ֵ

  RemoveNoise removeNoise;
  removeNoise.LessConnectedRegionRemove(image_vPolarEdgeInverseOSTU, 250);

  IplImage *image_vClose = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_OPEN,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_CLOSE,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_CLOSE,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_OPEN,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_CLOSE,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_CLOSE,4);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_ERODE,2);
  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_DILATE,3);
//  cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_DILATE,3);

  cvLogPolar(image_vClose, image_vBiPolar,
             cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,
             CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);
  cvThreshold(image_vBiPolar, image_vBiPolar, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);  //�ɿ��ǲ��ô����ֵ

  cvReleaseImage(&image_vPolar);
  cvReleaseImage(&image_vPolarEdge);
  cvReleaseImage(&image_vPolarEdgeInverse);
  cvReleaseImage(&image_vPolarEdgeInverseOSTU);
  cvReleaseImage(&image_vClose);
}
bool ftcc() {
  IplImage* image_vBiPolarContour = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  cvZero(image_vBiPolarContour);
  CvPoint contourStartPoint = cvPoint(-1, -1);
  bool contourStartFlag = false;
  for (int y = 0; y < image_vBiPolar->height; y++) {
    for (int x = 0; x < image_vBiPolar->width; x++) {
      int pixel_value = cvGetReal2D(image_vBiPolar, y, x);
      if (pixel_value > 200) {
        cvSetReal2D(image_vBiPolarContour, y, x, pixel_value);
        contourStartPoint.x = x;
        contourStartPoint.y = y;
        contourStartFlag = true;
        break;
      }
    }
    if (contourStartFlag) {
      break;
    }
  }
  if (contourStartPoint.x < 0) {
    isDetectTongue = false;
    return isDetectTongue;
  }

  for (int yy = contourStartPoint.y; yy < image_vBiPolar->height; yy++) {
    int r = 1;
    bool cicle_flag = false;
    while ((r < contourStartPoint.x) ||
           (r < image_vBiPolar->width - contourStartPoint.x - 1) ||
           (r < image_vBiPolar->height - contourStartPoint.y - 1)) {
      double PI = 3.14159265;
      int angle = 175;
      while (angle > 0) {
        int x, y;
        double rad = (angle * PI) / 180;
        x = contourStartPoint.x + cvRound(r * cos(rad));
        y = contourStartPoint.y + cvRound(r * sin(rad));
        if ((x >= 0) && (x < image_vBiPolar->width) && (y >= 0) &&
            (y < image_vBiPolar->height)) {
          if ((y > contourStartPoint.y) &&
              (cvGetReal2D(image_vBiPolar, y, x) > 200)) {
            contourStartPoint.x = x;
            contourStartPoint.y = y;
            cicle_flag = true;
            break;
          }
        }
        angle -= 5;
      }
      if (cicle_flag) {
        break;
      }
      r++;
    }
    for (int x = contourStartPoint.x; x >= 0; x--) {
      if (cvGetReal2D(image_vBiPolar, contourStartPoint.y, x) < 100) {
        contourStartPoint.x = x + 1;
        break;
      }
    }
    cvSetReal2D(image_vBiPolarContour, contourStartPoint.y, contourStartPoint.x,
                255);
    if (cicle_flag) {
      yy = contourStartPoint.y - 1;
    }
  }

  contourStartFlag = false;
  for (int y = image_vBiPolar->height - 1; y >= 0; y--) {
    for (int x = 0; x < image_vBiPolar->width; x++) {
      int pixel_value = cvGetReal2D(image_vBiPolar, y, x);
      if (pixel_value > 200) {
        cvSetReal2D(image_vBiPolarContour, y, x, pixel_value);
        contourStartPoint.x = x;
        contourStartPoint.y = y;
        contourStartFlag = true;
        break;
      }
    }
    if (contourStartFlag) {
      break;
    }
  }

  if (contourStartPoint.x < 0) {
    isDetectTongue = false;
    return isDetectTongue;
  }

  for (int yy = contourStartPoint.y; yy >= 0; yy--) {
    int r = 1;
    bool cicle_flag = false;
    while ((r < contourStartPoint.x) /*&&*/ ||
           (r < image_vBiPolar->width - contourStartPoint.x - 1) /*&&*/ ||
           (r < contourStartPoint.y)) {
      double PI = 3.14159265;
      int angle = -175;
      while (angle < 0) {
        int x, y;
        double rad = (angle * PI) / 180;
        x = contourStartPoint.x + cvRound(r * cos(rad));
        y = contourStartPoint.y + cvRound(r * sin(rad));
        if ((x >= 0) && (x < image_vBiPolar->width) && (y >= 0) &&
            (y < image_vBiPolar->height)) {
          if ((y < contourStartPoint.y) &&
              (cvGetReal2D(image_vBiPolar, y, x) > 200)) {
            contourStartPoint.x = x;
            contourStartPoint.y = y;
            cicle_flag = true;
            break;
          }
        }
        angle += 5;
      }
      if (cicle_flag) {
        break;
      }
      r++;
    }
    for (int x = contourStartPoint.x; x >= 0; x--) {
      if (cvGetReal2D(image_vBiPolar, contourStartPoint.y, x) < 100) {
        contourStartPoint.x = x + 1;
        break;
      }
    }
    cvSetReal2D(image_vBiPolarContour, contourStartPoint.y, contourStartPoint.x,
                255);
    if (cicle_flag) {
      yy = contourStartPoint.y + 1;
    }
  }
  for (int y = 0; y < image_vBiPolarContour->height; y++) {
    bool flag = false;
    int x;
    for (x = 0; x < image_vBiPolarContour->width; x++) {
      if (cvGetReal2D(image_vBiPolarContour, y, x) > 200) {
        flag = true;
        break;
      }
    }
    if (flag) {
      for (x = x + 1; x < image_vBiPolarContour->width; x++) {
        cvSetReal2D(image_vBiPolarContour, y, x, 0);
      }
    }
  }
  bool flag_top = false;
  CvPoint point_top;
  for (int y = 0; y < image_vBiPolarContour->height; y++) {
    for (int x = 0; x < image_vBiPolarContour->width; x++) {
      int pixel_value = cvGetReal2D(image_vBiPolarContour, y, x);
      if (pixel_value > 200) {
        cvSetReal2D(image_vBiPolarContour, 0, x, 255);
        point_top.x = x;
        point_top.y = 0;
        flag_top = true;
        break;
      }
    }
    if (flag_top) {
      break;
    }
  }

  bool flag_down = false;
  CvPoint point_down;
  for (int y = image_vBiPolarContour->height - 1; y >= 0; y--) {
    for (int x = image_vBiPolarContour->width - 1; x >= 0; x--) {
      int pixel_value = cvGetReal2D(image_vBiPolarContour, y, x);
      if (pixel_value > 200) {
        cvSetReal2D(image_vBiPolarContour, image_vBiPolarContour->height - 1, x,
                    255);
        point_down.x = x;
        point_down.y = image_vBiPolarContour->height - 1;
        flag_down = true;
        break;
      }
    }
    if (flag_down) {
      break;
    }
  }

  int flag_pre = 0, flag_now = 0;
  int positionY_pre = 0, positionY_now = 0;
  for (int y = 1; y < image_vBiPolarContour->height - 1; y++) {
    int y1 = 0, y2 = 0, x1 = 0, x2 = 0;
    for (int x = 0; x < image_vBiPolarContour->width; x++) {
      if (cvGetReal2D(image_vBiPolarContour, y, x) > 200) {
        y1 = y;
        x1 = x;
      }
    }
    bool flag = false;
    for (int i = 1; i < /*10*/ image_vBiPolarContour->height - 1 - y1; i++) {
      for (int x = 0; x < image_vBiPolarContour->width; x++) {
        if (cvGetReal2D(image_vBiPolarContour, y + i, x) > 200) {
          y2 = y + 1;
          x2 = x;
          flag = true;
          break;
        }
      }
      if (flag) {
        break;
      }
    }

    if (x2 - x1 > 5) {
      flag_now = 1;
      positionY_now = y2;
    } else if (x2 - x1 < -5) {
      flag_now = -1;
      positionY_now = y2;
    }
    if ((flag_pre > 0) && (flag_now < 0) &&
        (positionY_now - positionY_pre <
         image_vBiPolarContour->height * 1 / 10)) {
      for (int yy = positionY_pre; yy < positionY_now; yy++) {
        for (int xx = 0; xx < image_vBiPolarContour->width; xx++) {
          if (cvGetReal2D(image_vBiPolarContour, yy, xx) > 200) {
            cvSetReal2D(image_vBiPolarContour, yy, xx, 0);
          }
        }
      }
      flag_pre = flag_now;
      positionY_pre = positionY_now;
    } else {
      flag_pre = flag_now;
      positionY_pre = positionY_now;
    }
  }

  cvZero(image_vBiPolarcftt);
  cftt(image_vBiPolarContour, image_vBiPolarcftt, point_top, point_down);

  cvReleaseImage(&image_vBiPolarContour);

  return isDetectTongue;
}
void cftt(IplImage* image, IplImage* imageResult, CvPoint point_top,
          CvPoint point_down) {
  int distance = cvRound(
      sqrt((point_down.x - point_top.x) * (point_down.x - point_top.x) +
           (point_down.y - point_top.y) * (point_down.y - point_top.y)));
  int Td = image->height / 20;
  if (distance > Td) {
    CvPoint point_mid;
    point_mid.x = (point_top.x + point_down.x) / 2;
    point_mid.y = (point_top.y + point_down.y) / 2;
    int x_left = point_mid.x;
    int x_right = point_mid.x;
    while ((x_left >= 0) || (x_right <= image->width - 1)) {
      if (((x_left == point_top.x) && (point_mid.y == point_top.y)) ||
          ((x_left == point_down.x) && (point_mid.y == point_down.y)) ||
          ((x_right == point_top.x) && (point_mid.y == point_top.y)) ||
          ((x_right == point_down.x) &&
           (point_mid.y == point_down.y)))  //��ֹ����ݹ���ѭ��
      {
        point_mid.x = (point_top.x + point_down.x) / 2;
        point_mid.y = (point_top.y + point_down.y) / 2;
        break;
      }

      if (x_left >= 0) {
        if (cvGetReal2D(image, point_mid.y, x_left) > 200) {
          point_mid.x = x_left;
          break;
        }
      }
      if (x_right <= image->width - 1) {
        if (cvGetReal2D(image, point_mid.y, x_right) > 200) {
          point_mid.x = x_right;
          break;
        }
      }
      x_left--;
      x_right++;
    }
    if ((x_left < 0) && (x_right > image->width - 1)) {
      CvPoint pointNear_top, pointNear_down;
      pointNear_top.x = point_mid.x;
      pointNear_top.y = point_mid.y;
      pointNear_down.x = point_mid.x;
      pointNear_down.y = point_mid.y;
      bool flag_top = false, flag_down = false;
      for (int y = point_mid.y; y >= 0; y--) {
        int x_left = point_mid.x;
        int x_right = point_mid.x;
        while ((x_left >= 0) || (x_right <= image->width - 1)) {
          if (x_left >= 0) {
            if (cvGetReal2D(image, y, x_left) > 200) {
              pointNear_top.x = x_left;
              pointNear_top.y = y;
              flag_top = true;
              break;
            }
          }
          if (x_right <= image->width - 1) {
            if (cvGetReal2D(image, y, x_right) > 200) {
              pointNear_top.x = x_right;
              pointNear_top.y = y;
              flag_top = true;
              break;
            }
          }
          x_left--;
          x_right++;
        }
        if (flag_top) {
          break;
        }
      }

      for (int y = point_mid.y; y <= image->width - 1; y++) {
        int x_left = point_mid.x;
        int x_right = point_mid.x;
        while ((x_left >= 0) || (x_right <= image->width - 1)) {
          if (x_left >= 0) {
            if (cvGetReal2D(image, y, x_left) > 200) {
              pointNear_down.x = x_left;
              pointNear_down.y = y;
              flag_down = true;
              break;
            }
          }
          if (x_right <= image->width - 1) {
            if (cvGetReal2D(image, y, x_right) > 200) {
              pointNear_down.x = x_right;
              pointNear_down.y = y;
              flag_down = true;
              break;
            }
          }
          x_left--;
          x_right++;
        }
        if (flag_down) {
          break;
        }
      }
      if (pointNear_down.y != pointNear_top.y) {
        point_mid.x = cvRound(pointNear_top.x +
                              (point_mid.y - pointNear_top.y) *
                                  (pointNear_down.x - pointNear_top.x) /
                                  (pointNear_down.y - pointNear_top.y));
      } else {
        point_mid.x = pointNear_top.x;
      }
      cvSetReal2D(image, point_mid.y, point_mid.x, 255);
    }

    cftt(image, imageResult, point_top, point_mid);
    cftt(image, imageResult, point_mid, point_down);
  } else {
    if (distance > 0) {
      cvLine(imageResult, point_top, point_down, cvScalar(255), 1, 8);
    }
  }
}
void ctmm() {
  int M = 40;
  IplImage* image_vBiPolarFill = cvCreateImage(
      cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);
  image_vBiPolarFill = cvCloneImage(image_vBiPolarcftt);
  for (int y = 0; y < image_vBiPolarFill->height - 1; y++) {
    for (int x = 0; x < image_vBiPolarFill->width - 1; x++) {
      if (cvGetReal2D(image_vBiPolarFill, y, x) < 100) {
        cvSetReal2D(image_vBiPolarFill, y, x, 255);
      } else {
        break;
      }
    }
  }

  cvLogPolar(image_vBiPolarFill, image_vMask,
             cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,
             CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
  cvThreshold(image_vMask, image_vMask, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);

  IplImage* image_vOSTU;
  image_vOSTU = cvCloneImage(image_v);
  cvThreshold(image_vOSTU, image_vOSTU, 80, 255,
              CV_THRESH_BINARY /* | CV_THRESH_OTSU*/);
  for (int y = 0; y < image_vMask->height; y++) {
    for (int x = 0; x < image_vMask->width; x++) {
      if (cvGetReal2D(image_vOSTU, y, x) < 100) {
        cvSetReal2D(image_vMask, y, x, 0);
      }
    }
  }

  cvMorphologyEx(image_vMask, image_vMask, NULL, NULL, CV_MOP_CLOSE, 4);

  cvReleaseImage(&image_vBiPolarFill);
}
void sgtt() {
  image_tongue = cvCloneImage(image_tongueROI);
  for (int y = 0; y < image_vMask->height - 1; y++) {
    for (int x = 0; x < image_vMask->width - 1; x++) {
      if (cvGetReal2D(image_vMask, y, x) < 100) {
        cvSet2D(image_tongue, y, x, cvScalar(0, 0, 0));
      }
    }
  }
}
IplImage* gtmm() { return image_vMask; }
IplImage* gtii() { return image_tongue; }
bool ccmm() {
  CvRect tongueRect = cvRect(0, 0, 0, 0);
  IplImage* image_TongueContour = cvCreateImage(cvGetSize(image_vMask), 8, 1);
  cvZero(image_TongueContour);
  dtmc(image_TongueContour, tongueRect);

  int center_x, center_y;
  center_x = tongueRect.x + tongueRect.width / 2;
  center_y = tongueRect.y + tongueRect.height / 2;
  if (center_x == 0 || center_y == 0) {
    isDetectTongue = false;
    return isDetectTongue;
  }

  int up, down, left, right;
  up = center_y - 1;
  down = center_y + 1;
  left = center_x - 1;
  right = center_x + 1;
  bool isStopUp = false;
  bool isStopDown = false;
  bool isStopLeft = false;
  bool isStopRight = false;

  while (!(isStopUp && isStopDown && isStopLeft && isStopRight)) {
    if (!isStopUp) {
      for (int x = left; x <= right; x++) {
        if (cvGetReal2D(image_TongueContour, up, x) > 100) {
          isStopUp = true;
          break;
        }
      }
      up--;
      if (up <= 0) {
        isStopUp = true;
      }
    }

    if (!isStopDown) {
      for (int x = left; x <= right; x++) {
        if (cvGetReal2D(image_TongueContour, down, x) > 100) {
          isStopDown = true;
          break;
        }
      }
      down++;
      if (down >= image_TongueContour->height - 1) {
        isStopDown = true;
      }
    }
    if (!isStopLeft) {
      for (int y = up; y <= down; y++) {
        if (cvGetReal2D(image_TongueContour, y, left) > 100) {
          isStopLeft = true;
          break;
        }
      }
      left--;
      if (left <= 0) {
        isStopLeft = true;
      }
    }

    if (!isStopRight) {
      for (int y = up; y <= down; y++) {
        if (cvGetReal2D(image_TongueContour, y, right) > 100) {
          isStopRight = true;
          break;
        }
      }
      right++;
      if (right >= image_TongueContour->width - 1) {
        isStopRight = true;
      }
    }
  }
  if (up <= tongueRect.y + tongueRect.height / 5) {
    up = tongueRect.y + tongueRect.height / 5;
  }
  int width1 = right - left;
  left += width1 / 10;
  right -= width1 / 10;

  CvRect rectMIR = cvRect(left, up, right - left, down - up);

  cvSetImageROI(image_tongue, rectMIR);
  image_MIRTongue = cvCreateImage(cvGetSize(image_tongue), 8, 3);
  cvCopy(image_tongue, image_MIRTongue);
  cvResetImageROI(image_tongue);
  cvReleaseImage(&image_TongueContour);

  return isDetectTongue;
}
void dtmc(IplImage*& image_TongueContour1, CvRect& tongueRect1) {
  IplImage* image_vMaskCpy = cvCloneImage(image_vMask);
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(image_vMaskCpy, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  int areaMax = 0;
  CvSeq* pContourTongue = NULL;
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);
    if (area > areaMax) {
      areaMax = area;
      tongueRect1 = rect;
      pContourTongue = pContour;
    }
  }
  if (areaMax != 0) {
    cvDrawContours(image_TongueContour1, pContourTongue, CV_RGB(255, 255, 255),
                   CV_RGB(255, 255, 255), 0, 1);
  }

  cvReleaseMemStorage(&pStorage);
  cvReleaseImage(&image_vMaskCpy);
}
IplImage* gtim() { return image_MIRTongue; }

void tongueCleanup() {
    cvReleaseMemStorage(&storageS);
    cvReleaseImage(&imageSrcS);
    cvReleaseImage(&inputImageS);
    cvReleaseImage(&image_tongueROI);
    cvReleaseImage(&image_v);
    cvReleaseImage(&image_vBiPolar);
    cvReleaseImage(&image_vBiPolarcftt);
    cvReleaseImage(&image_vMask);
    cvReleaseImage(&image_tongue);
    cvReleaseImage(&image_MIRTongue);
    cvReleaseImage(&I);
    cvReleaseImage(&p);
    cvReleaseImage(&win_mat);
    cvReleaseImage(&inputImagecucao);
    cvReleaseImage(&image_sdv);
    cvReleaseImage(&image_roughness);
    cvReleaseImage(&q);
    cvReleaseImage(&inputImageandu);
    cvReleaseImage(&darkImageBi);
    cvReleaseImage(&inputImageshuiliu);
    cvReleaseImage(&inputRoughness);
    cvReleaseImage(&L);
    cvReleaseImage(&tongueImage);
    cvReleaseImage(&tongueMask);
    cvReleaseImage(&tongueCoat);
    cvReleaseImage(&tongueNature);
    cvReleaseImage(&tongueCoatImage);
    cvReleaseImage(&tongueNatureImage);
    cvReleaseImage(&tongueCoatImagecc);

    cascade_nameS = NULL;
    isDetectTongue = false;
    switchHaarTongue = 0;
    rlvbo = 0;
    eps = 0.0;
    N = 0;
    rcucao = 0;
    rdu = 0;
    T = 0;
    g = 0;
    cascade_nameS = NULL;
    panClassifierPathName = NULL;
    feature[0] = 0;
    feature[1] = 0;
    bohouClassifierPathName = NULL;
    ccolorClassifierPathName = NULL;
}
