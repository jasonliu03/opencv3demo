#include "faceDignosis.h"
#include "GMM.h"
#include "RemoveNoise.h"
#include "../jni/include/Global.h"

char const* cascade_name = 0;
CvRect lipRect;
CvMemStorage* storageL;
CvHaarClassifierCascade* cascadeL;
double roiScale2 = 1.0;
char const* cascade_nameL = 0;
IplImage* imageSrcL;
CvMemStorage* storage;
CvHaarClassifierCascade* cascade;
IplImage* imageSrc;
CvRect faceRect;
IplImage* inputImage;
IplImage* face;
IplImage* faceImage;
IplImage* faceImage_Lab;
IplImage* faceImage_Gauss;
IplImage* faceSkinImage;
IplImage* faceSkinMask;
int nGMM;
GMM* mComplexionGMM;
int switchHaarFace;
bool isDetectFace;
char const* colorClassifierPathName = "svm_facecomplexion.xml";
char const* glossModelPathName = "facegloss_model.xml";
#define DECREASE -1
#define INCREASE 1
IplImage* faceZ;
IplImage* faceDwn;
IplImage* faceDwn_Gauss;
IplImage* lipExtractMask;
IplImage* lipExtractImage;
IplImage* lipRawMask;
IplImage* lipImage;
const int nGMMZ = 5;
GMM* mComplexionGMMZ;
char const* colorClassifierPathNameZ = "svm_lipcolor.xml";
int cptz(const CvScalar lipColor) {
	int response = 2;
	float feature[3];
	feature[0] = lipColor.val[0];
	feature[1] = lipColor.val[1];
	feature[2] = lipColor.val[2];
	if (feature[0] > 65 && feature[0] <= 120) {
		response = 1;
	}
	else {
		Mat testDataMat(1, 3, CV_32FC1, feature);

        Mat responses;
        Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(colorClassifierPathNameZ); //读取模型
        svm->predict(testDataMat, responses);
        responses.convertTo(responses,CV_32S);
        response = responses.at<int>(0,0);
	}
	return response;
}
void lsgi(const IplImage* face) {
  dysz(face);
  mComplexionGMMZ = NULL;
  faceDwn = NULL;
  faceDwn_Gauss = NULL;
  lipExtractMask = NULL;
  lipExtractImage = NULL;
  lipRawMask = NULL;
  lipImage = NULL;
}
void dysz(const IplImage* inputImage) {
  int size;
  size = (inputImage->height > inputImage->width) ? inputImage->height
                                                  : inputImage->width;
  double type = ((double)size) / 600.0;
  if (type <= 1) {
    faceZ = cvCloneImage(inputImage);
  } else {
    faceZ = cvCreateImage(cvSize(cvRound(inputImage->width / type),
                                 cvRound(inputImage->height / type)),
                          inputImage->depth, inputImage->nChannels);
    cvResize(inputImage, faceZ);
  }
}
bool prodp() {
	bcmm();
	ccpm();
	CvRect rect_Lip = cvRect(0, 0, 0, 0);
	IplImage* image_LipBi = NULL;
	ldi(faceZ);
	if (detectLip()) {
		rect_Lip = gtrl();
		lipFenGe(image_LipBi, rect_Lip);
		opmm(image_LipBi, rect_Lip);
		lcet(rect_Lip.width / 10);
		// cvSaveImage("lipExtractImage.jpg", lipExtractImage);
		// cvSaveImage("lipImage.jpg", lipImage);
		cvReleaseImage(&image_LipBi);
		return true;
	}
	else {
		return false;
	}
}
void ldi(IplImage* imageSrc2) {
	imageSrcL = cvCloneImage(imageSrc2);
	if (storageL) {
		cvReleaseMemStorage(&storageL);
		storageL = 0;
	}
	cascadeL = 0;
}
bool detectLip() {

	cascadeL = (CvHaarClassifierCascade*)cvLoad(cascade_nameL, 0, 0, 0);
	storageL = cvCreateMemStorage(0);

	double scale = 1.0;
	IplImage* gray =
		cvCreateImage(cvSize(imageSrcL->width, imageSrcL->height), 8, 1);
	IplImage* small_img =
		cvCreateImage(cvSize(cvRound(imageSrcL->width / scale),
			cvRound(imageSrcL->height / scale)), 8, 1);
	cvCvtColor(imageSrcL, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvClearMemStorage(storageL);

	CvRect *r, maxR;
	int rSize, maxSize = 0;
	int cnt = 0;
	if (cascadeL) {
		CvSeq* lips = cvHaarDetectObjects(small_img, cascadeL, storageL, 1.1, 2,
			CV_HAAR_FIND_BIGGEST_OBJECT |
			CV_HAAR_DO_ROUGH_SEARCH |
			CV_HAAR_DO_CANNY_PRUNING,
			cvSize(10, 10));         // Ҳ������ 40*40 
		cnt = lips->total;
		for (int i = 0; i < (lips ? lips->total : 0); i++) {
			r = (CvRect*)cvGetSeqElem(lips, i);
			rSize = r->width * r->height;
			if (rSize > maxSize) {
				maxSize = rSize;
				maxR = *r;
				lipRect = *r;
			}
		}
	}
	else {
		cout << "Loading haarcascade_lip.xml is wrong !!" << endl;
	}

	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	if (storageL) {
		cvReleaseMemStorage(&storageL);
		storageL = 0;
	}
	if (cnt == 0) {
		maxR.x = 0;
		maxR.y = 0;
		maxR.width = 0;
		maxR.height = 0;
		return false;
	}
	else {
		int compareLength = 200;
		int roiLength = lipRect.width < lipRect.height ? lipRect.width : lipRect.height;
		roiScale2 = compareLength * 1.0 / roiLength;
		lipRect.x = lipRect.x * scale;
		lipRect.y = lipRect.y * scale;
		lipRect.width = lipRect.width * scale;
		lipRect.height = lipRect.height * scale;
		return true;
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
}
CvRect gtrl() {
	return lipRect;
}
void lipFenGe(IplImage*& image_LipBi, CvRect rect_Lip) {
	IplImage* faceZ_Gray = cvCreateImage(cvGetSize(faceZ), 8, 1);
	cvCvtColor(faceZ, faceZ_Gray, CV_RGB2GRAY);
	cvSetImageROI(faceZ_Gray, rect_Lip);
	IplImage* image_Lip = cvCreateImage(cvGetSize(faceZ_Gray), 8, 1);
	cvCopy(faceZ_Gray, image_Lip);
	cvResetImageROI(faceZ_Gray);
	cvSetImageROI(faceDwn_Gauss, rect_Lip);
	IplImage* image_Gauss =
		cvCreateImage(cvGetSize(faceDwn_Gauss), faceDwn_Gauss->depth, 1);
	cvCopy(faceDwn_Gauss, image_Gauss);
	cvResetImageROI(faceDwn_Gauss);
	image_LipBi = cvCloneImage(image_Lip);
	int s1, s2, sd;
	s1 = llrf(image_Lip);
	int it = 0;
	int sdFlagPre, sdFlagCur;
	int sdItPre = it, sdItCur = it;
	int sdPre, sdCur;
	int continuousIncreaseCnt = 0;
	int continuousDecreaseCnt = 0;
	int nLipRefine;
	double iterate_Threshold[16] = { 0.3,   0.2,   0.1,   0.09, 0.08, 0.07,
		0.06,  0.05,  0.04,  0.03, 0.02, 0.01,
		0.009, 0.007, 0.005, 0.003 };
	for (int i = it + 1; i < 16; i++) {
		for (int y = 0; y < image_Lip->height; y++) {
			for (int x = 0; x < image_Lip->width; x++) {
				if (cvGetReal2D(image_Gauss, y, x) >= iterate_Threshold[i]) {
					cvSetReal2D(image_Lip, y, x, 0);
				}
			}
		}
		s2 = llrf(image_Lip);
		sd = s1 - s2;
		s1 = s2;
		if (i == it + 1) {
			sdPre = sd;
			sdFlagPre = 0;
			sdItPre = i;
			sdItCur = i;
		}
		else {
			sdCur = sd;
			sdItCur = i;
			if (sdCur - sdPre > 0) {
				sdFlagCur = INCREASE;
				++continuousIncreaseCnt;
				continuousDecreaseCnt = 0;
			}
			else {
				sdFlagCur = DECREASE;
				continuousIncreaseCnt = 0;
				++continuousDecreaseCnt;
			}
			if (continuousIncreaseCnt >= 3 || continuousDecreaseCnt >= 3) {
				break;
			}
			if ((sdFlagPre == DECREASE) && (sdFlagCur == INCREASE)) {
				break;
			}
			else {
				sdPre = sdCur;
				sdItPre = sdItCur;
				sdFlagPre = sdFlagCur;
			}
		}
		{
			IplImage* faceDownGrayRefine = cvCreateImage(cvGetSize(faceZ), 8, 1);
			cvZero(faceDownGrayRefine);
			for (int y = 0; y < image_Lip->height; y++) {
				for (int x = 0; x < image_Lip->width; x++) {
					int pixel = cvGetReal2D(image_Lip, y, x);
					cvSetReal2D(faceDownGrayRefine, rect_Lip.y + y, rect_Lip.x + x,
						pixel);
				}
			}
			nLipRefine = lpdt(faceDownGrayRefine);
			if (nLipRefine == 0) {
				if (sdItPre > 0) {
					sdItPre--;
				}
				sdItCur = (it + sdItPre) / 2;
				break;
			}
			cvReleaseImage(&faceDownGrayRefine);
		}
	}
	double T = 0.0;
	if (nLipRefine == 0) {
		T = (iterate_Threshold[sdItPre] +
			iterate_Threshold[sdItCur] /*+iterate_Threshold[it]*/) /
			2;
	}
	else {
		T = (iterate_Threshold[sdItPre] +
			iterate_Threshold[sdItPre] /*+iterate_Threshold[it]*/) /
			2;
	}
	for (int y = 0; y < image_LipBi->height; y++) {
		for (int x = 0; x < image_LipBi->width; x++) {
			if (cvGetReal2D(image_Gauss, y, x) >= T) {
				cvSetReal2D(image_LipBi, y, x, 0);
			}
		}
	}
	cvSmooth(image_LipBi, image_LipBi, CV_MEDIAN);
	ilmf(image_LipBi);
	cvReleaseImage(&faceZ_Gray);
	cvReleaseImage(&image_Lip);
	cvReleaseImage(&image_Gauss);
}
CvScalar elcf() {
  IplImage* lipImageGray = cvCreateImage(cvGetSize(lipImage), 8, 1);
  cvCvtColor(lipImage, lipImageGray, CV_BGR2GRAY);
  int i_dark, i_bright;
  double rate_dark = 0.2, rate_bright = 0.02;
  cadbt(lipRawMask, lipImageGray, i_dark, i_bright, rate_dark, rate_bright, 0);
  for (int y = 0; y < lipRawMask->height; ++y) {
    for (int x = 0; x < lipRawMask->width; ++x) {
      if (cvGetReal2D(lipImageGray, y, x) < i_dark ||
          cvGetReal2D(lipImageGray, y, x) > i_bright) {
        cvSetReal2D(lipRawMask, y, x, 0);
      }
    }
  }
  cvErode(lipRawMask, lipRawMask);
  RemoveNoise rn;
  rn.LessConnectedRegionRemove(lipRawMask,
                               lipRawMask->height * lipRawMask->width / 30);
  IplImage* lipImage_Lab = cvCreateImage(cvGetSize(lipImage), 8, 3);
  cvCvtColor(lipImage, lipImage_Lab, CV_BGR2Lab);
  CvScalar lipColorFeature = cvAvg(lipImage_Lab, lipRawMask);
  cvReleaseImage(&lipImageGray);
  cvReleaseImage(&lipImage_Lab);
  return lipColorFeature;
}
void bcmm() {
  int len;
  len = (faceZ->height > faceZ->width) ? faceZ->height : faceZ->width;
  double type = ((double)len) / 60.0;
  IplImage* faceScale = cvCreateImage(
      cvSize(cvRound(faceZ->width / type), cvRound(faceZ->height / type)),
      faceZ->depth, faceZ->nChannels);
  cvResize(faceZ, faceScale);
  IplImage* faceImageScaleGray = cvCreateImage(cvGetSize(faceScale), 8, 1);
  cvCvtColor(faceScale, faceImageScaleGray, CV_BGR2GRAY);
  IplImage* faceScaleEllipseMask = cvCreateImage(cvGetSize(faceScale), 8, 1);
  cvZero(faceScaleEllipseMask);
  CvPoint center = cvPoint(cvRound(faceScaleEllipseMask->width * 0.5),
                           cvRound(faceScaleEllipseMask->height * 0.5));
  CvSize size = cvSize(cvRound(faceScaleEllipseMask->width * 0.36),
                       cvRound(faceScaleEllipseMask->height * 0.50));
  cvEllipse(faceScaleEllipseMask, center, size, 0, 0, 360, cvScalar(255),
            CV_FILLED);
  int i_dark, i_bright;
  double rate_dark = 0.2, rate_bright = 0;
  float lowPart = 0.65;
  cadbt(faceScaleEllipseMask, faceImageScaleGray, i_dark, i_bright, rate_dark,
        rate_bright, lowPart);
  for (int y = faceScaleEllipseMask->height * lowPart;
       y < faceScaleEllipseMask->height; ++y) {
    for (int x = 0; x < faceScaleEllipseMask->width; ++x) {
      if (cvGetReal2D(faceImageScaleGray, y, x) < i_dark) {
        cvSetReal2D(faceScaleEllipseMask, y, x, 0);
      }
    }
  }
  cvErode(faceScaleEllipseMask, faceScaleEllipseMask);
  IplImage* faceImageScale_Lab = NULL;
  faceImageScale_Lab = cvCreateImage(cvGetSize(faceScale), 8, 3);
  cvCvtColor(faceScale, faceImageScale_Lab, CV_BGR2Lab);
  uint cnt = 0, nrows = 0;
  for (int y = 0; y < faceScaleEllipseMask->height; y++) {
    for (int x = 0; x < faceScaleEllipseMask->width; x++) {
      if (y <= faceScaleEllipseMask->height * 2 / 3) {
        nrows++;
      } else {
        if (cvGetReal2D(faceScaleEllipseMask, y, x) > 200) nrows++;
      }
    }
  }
  double** data;
  data = (double**)malloc(nrows * sizeof(double*));
  for (int i = 0; i < nrows; i++) data[i] = (double*)malloc(3 * sizeof(double));
  for (int y = 0; y < faceScaleEllipseMask->height; y++) {
    for (int x = 0; x < faceScaleEllipseMask->width; x++) {
      if (y <= faceScaleEllipseMask->height * 2 / 3) {
        data[cnt][0] = cvGet2D(faceImageScale_Lab, y, x).val[0];
        data[cnt][1] = cvGet2D(faceImageScale_Lab, y, x).val[1];
        data[cnt++][2] = cvGet2D(faceImageScale_Lab, y, x).val[2];
      } else {
        if (cvGetReal2D(faceScaleEllipseMask, y, x) > 200) {
          data[cnt][0] = cvGet2D(faceImageScale_Lab, y, x).val[0];
          data[cnt][1] = cvGet2D(faceImageScale_Lab, y, x).val[1];
          data[cnt++][2] = cvGet2D(faceImageScale_Lab, y, x).val[2];
        }
      }
    }
  }
  mComplexionGMMZ = new GMM(nGMMZ);
  mComplexionGMMZ->Build(data, nrows);
  for (int i = 0; i < nrows; i++) free(data[i]);
  free(data);
  cvReleaseImage(&faceScale);
  cvReleaseImage(&faceImageScaleGray);
  cvReleaseImage(&faceScaleEllipseMask);
  cvReleaseImage(&faceImageScale_Lab);
}
void ccpm() {
  /*cvSetImageROI(faceZ,
                cvRect(0, faceZ->height / 2, faceZ->width, faceZ->height / 2));
  faceDwn = cvCreateImage(cvGetSize(faceZ), 8, 3);
  cvCopy(faceZ, faceDwn);
  cvResetImageROI(faceZ);*/
  IplImage* faceDwn_Lab = NULL;
  faceDwn_Lab = cvCreateImage(cvGetSize(faceZ), 8, 3);
  cvCvtColor(faceZ, faceDwn_Lab, CV_BGR2Lab);
  faceDwn_Gauss = cvCreateImage(cvGetSize(faceZ), IPL_DEPTH_64F, 1);
  cvZero(faceDwn_Gauss);
  for (int y = 0; y < faceDwn_Gauss->height; y++) {
    for (int x = 0; x < faceDwn_Gauss->width; x++) {
      CvScalar pixel = cvGet2D(faceDwn_Lab, y, x);
      Color c(pixel.val[0], pixel.val[1], pixel.val[2]);
      float px = mComplexionGMMZ->p(c);
      cvSetReal2D(faceDwn_Gauss, y, x, px);
    }
  }
  cvNormalize(faceDwn_Gauss, faceDwn_Gauss, 1.0, 0.0, CV_C);
  cvSmooth(faceDwn_Gauss, faceDwn_Gauss, CV_GAUSSIAN, 5, 5);
  cvReleaseImage(&faceDwn_Lab);
}
void opmm(const IplImage* image_LipBi, const CvRect rect_Lip) {
  cvSetImageROI(faceZ, rect_Lip);
  lipImage = cvCreateImage(cvGetSize(faceZ), 8, 3);
  cvCopy(faceZ, lipImage);
  cvResetImageROI(faceZ);
  IplImage* lipBi = cvCloneImage(image_LipBi);
  int len;
  len =
      (lipImage->height > lipImage->width) ? lipImage->height : lipImage->width;
  double type = ((double)len) / 50.0;
  IplImage* lipImageScale = cvCreateImage(
      cvSize(cvRound(lipImage->width / type), cvRound(lipImage->height / type)),
      lipImage->depth, lipImage->nChannels);
  cvResize(lipImage, lipImageScale);
  IplImage* lipBiScale = cvCreateImage(
      cvSize(cvRound(lipBi->width / type), cvRound(lipBi->height / type)),
      lipBi->depth, lipBi->nChannels);
  cvResize(lipBi, lipBiScale);
  cvThreshold(lipBiScale, lipBiScale, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU);
  IplImage* lipScale_Lab = NULL;
  lipScale_Lab = cvCreateImage(cvGetSize(lipImageScale), 8, 3);
  cvCvtColor(lipImageScale, lipScale_Lab, CV_BGR2Lab);
  uint cnt1 = 0, nrows1 = 0;
  double** data1;
  for (int y = 0; y < lipBiScale->height; y++) {
    for (int x = 0; x < lipBiScale->width; x++) {
      if (cvGetReal2D(lipBiScale, y, x) > 200) {
        nrows1++;
      }
    }
  }
  data1 = (double**)malloc(nrows1 * sizeof(double*));
  for (int i = 0; i < nrows1; i++)
    data1[i] = (double*)malloc(3 * sizeof(double));
  for (int y = 0; y < lipBiScale->height; y++) {
    for (int x = 0; x < lipBiScale->width; x++) {
      if (cvGetReal2D(lipBiScale, y, x) > 200) {
        data1[cnt1][0] = cvGet2D(lipScale_Lab, y, x).val[0];
        data1[cnt1][1] = cvGet2D(lipScale_Lab, y, x).val[1];
        data1[cnt1++][2] = cvGet2D(lipScale_Lab, y, x).val[2];
      }
    }
  }
  GMM* mComplexionGMM1 = new GMM(3);
  mComplexionGMM1->Build(data1, nrows1);
  for (int i = 0; i < nrows1; i++) free(data1[i]);
  free(data1);
  uint cnt2 = 0, nrows2 = 0;
  double** data2;
  for (int y = 0; y < lipBiScale->height; y++) {
    for (int x = 0; x < lipBiScale->width; x++) {
      if (cvGetReal2D(lipBiScale, y, x) < 100) {
        nrows2++;
      }
    }
  }
  data2 = (double**)malloc(nrows2 * sizeof(double*));
  for (int i = 0; i < nrows2; i++)
    data2[i] = (double*)malloc(3 * sizeof(double));
  for (int y = 0; y < lipBiScale->height; y++) {
    for (int x = 0; x < lipBiScale->width; x++) {
      if (cvGetReal2D(lipBiScale, y, x) < 100) {
        data2[cnt2][0] = cvGet2D(lipScale_Lab, y, x).val[0];
        data2[cnt2][1] = cvGet2D(lipScale_Lab, y, x).val[1];
        data2[cnt2++][2] = cvGet2D(lipScale_Lab, y, x).val[2];
      }
    }
  }
  GMM* mComplexionGMM2 = new GMM(3);
  mComplexionGMM2->Build(data2, nrows2);
  for (int i = 0; i < nrows2; i++) free(data2[i]);
  free(data2);
  IplImage* lip_Lab = NULL;
  lip_Lab = cvCreateImage(cvGetSize(lipImage), 8, 3);
  cvCvtColor(lipImage, lip_Lab, CV_BGR2Lab);
  IplImage* lip_Gauss = cvCreateImage(cvGetSize(lip_Lab), IPL_DEPTH_64F, 1);
  cvZero(lip_Gauss);
  for (int y = 0; y < lip_Gauss->height; y++) {
    for (int x = 0; x < lip_Gauss->width; x++) {
      CvScalar pixel = cvGet2D(lip_Lab, y, x);
      Color c(pixel.val[0], pixel.val[1], pixel.val[2]);
      float px_f = mComplexionGMM1->p(c);
      float px_b = mComplexionGMM2->p(c);
      float px = px_b / (px_f + px_b);
      cvSetReal2D(lip_Gauss, y, x, px);
    }
  }
  cvNormalize(lip_Gauss, lip_Gauss, 1.0, 0.0, CV_C);
  cvSmooth(lip_Gauss, lip_Gauss, CV_GAUSSIAN, 5, 5);
  IplImage* lip_similar = cvCreateImage(cvGetSize(lip_Gauss), 8, 1);
  cvScale(lip_Gauss, lip_similar, 255);
  double biThreshold = cvThreshold(lip_similar, lip_similar, 200, 255,
                                   CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  RemoveNoise rn;
  rn.LessConnectedRegionRemove(lip_similar,
                               lip_similar->height * lip_similar->width / 20);
  lipRawMask = cvCloneImage(lip_similar);
  lipExtractMask = cvCreateImage(cvGetSize(faceZ), 8, 1);
  cvZero(lipExtractMask);
  for (int y = 0; y < lip_similar->height; y++) {
    for (int x = 0; x < lip_similar->width; x++) {
      int pixel = cvGetReal2D(lip_similar, y, x);
      cvSetReal2D(lipExtractMask, rect_Lip.y + y, rect_Lip.x + x, pixel);
    }
  }
  cvSmooth(lipExtractMask, lipExtractMask, CV_MEDIAN, 5);
  cvReleaseImage(&lipBi);
  cvReleaseImage(&lipImageScale);
  cvReleaseImage(&lipBiScale);
  cvReleaseImage(&lipScale_Lab);
  delete mComplexionGMM1;
  delete mComplexionGMM2;
  cvReleaseImage(&lip_Lab);
  cvReleaseImage(&lip_Gauss);
  cvReleaseImage(&lip_similar);
}
bool dtlr(IplImage*& image_LipBi, CvRect& rect_Lip) {
  IplImage* faceDwn_Gray = cvCreateImage(cvGetSize(faceDwn), 8, 1);
  cvZero(faceDwn_Gray);
  cvAddS(faceDwn_Gray, cvScalar(255), faceDwn_Gray);
  double iterate_Threshold[16] = {0.3,   0.2,   0.1,   0.09, 0.08, 0.07,
                                  0.06,  0.05,  0.04,  0.03, 0.02, 0.01,
                                  0.009, 0.007, 0.005, 0.003};
  CvRect lip_region = cvRect(0, 0, 0, 0);
  bool flag_Lip = false;
  for (int i = 6; i < 16; i++) {
    IplImage* faceDown_Gray2 = cvCloneImage(faceDwn_Gray);
    for (int y = 0; y < faceDwn_Gauss->height; y++) {
      for (int x = 0; x < faceDwn_Gauss->width; x++) {
        if (cvGetReal2D(faceDwn_Gauss, y, x) >= iterate_Threshold[i]) {
          cvSetReal2D(faceDown_Gray2, y, x, 0);
        }
      }
    }
    IplImage* faceDown_GrayT = cvCloneImage(faceDown_Gray2);
    lip_region = lple(faceDown_GrayT, flag_Lip);
    cvReleaseImage(&faceDown_GrayT);
    cvReleaseImage(&faceDown_Gray2);
    if (flag_Lip) {
      break;
    }
  }
  if (!flag_Lip) {
    for (int i = 5; i >= 0; i--) {
      IplImage* faceDown_Gray2 = cvCloneImage(faceDwn_Gray);
      for (int y = 0; y < faceDwn_Gauss->height; y++) {
        for (int x = 0; x < faceDwn_Gauss->width; x++) {
          if (cvGetReal2D(faceDwn_Gauss, y, x) >= iterate_Threshold[i]) {
            cvSetReal2D(faceDown_Gray2, y, x, 0);
          }
        }
      }
      IplImage* faceDown_GrayT = cvCloneImage(faceDown_Gray2);
      lip_region = lple(faceDown_GrayT, flag_Lip);
      cvReleaseImage(&faceDown_GrayT);
      cvReleaseImage(&faceDown_Gray2);
      if (flag_Lip) {
        break;
      }
    }
  }
  if (!flag_Lip) {
    return flag_Lip;
  }
  flag_Lip = false;
  int it;
  for (it = 0; it < 16; it = it + 1) {
    for (int y = 0; y < faceDwn_Gauss->height; y++) {
      for (int x = 0; x < faceDwn_Gauss->width; x++) {
        if (cvGetReal2D(faceDwn_Gauss, y, x) >= iterate_Threshold[it]) {
          cvSetReal2D(faceDwn_Gray, y, x, 0);
        }
      }
    }
    IplImage* faceDown_GrayT = cvCloneImage(faceDwn_Gray);
    rect_Lip = lpal(faceDown_GrayT, flag_Lip, lip_region);
    cvReleaseImage(&faceDown_GrayT);
    if (flag_Lip) {
      break;
    }
  }
  if (!flag_Lip) {
    return flag_Lip;
  }
  cvSetImageROI(faceDwn_Gray, rect_Lip);
  IplImage* image_Lip = cvCreateImage(cvGetSize(faceDwn_Gray), 8, 1);
  cvCopy(faceDwn_Gray, image_Lip);
  cvResetImageROI(faceDwn_Gray);
  cvSetImageROI(faceDwn_Gauss, rect_Lip);
  IplImage* image_Gauss =
      cvCreateImage(cvGetSize(faceDwn_Gauss), faceDwn_Gauss->depth, 1);
  cvCopy(faceDwn_Gauss, image_Gauss);
  cvResetImageROI(faceDwn_Gauss);
  image_LipBi = cvCloneImage(image_Lip);
  int s1, s2, sd;
  s1 = llrf(image_Lip);
  int sdFlagPre, sdFlagCur;
  int sdItPre = it, sdItCur = it;
  int sdPre, sdCur;
  int continuousIncreaseCnt = 0;
  int continuousDecreaseCnt = 0;
  int nLipRefine;
  for (int i = it + 1; i < 16; i++) {
    for (int y = 0; y < image_Lip->height; y++) {
      for (int x = 0; x < image_Lip->width; x++) {
        if (cvGetReal2D(image_Gauss, y, x) >= iterate_Threshold[i]) {
          cvSetReal2D(image_Lip, y, x, 0);
        }
      }
    }
    s2 = llrf(image_Lip);
    sd = s1 - s2;
    s1 = s2;
    if (i == it + 1) {
      sdPre = sd;
      sdFlagPre = 0;
      sdItPre = i;
      sdItCur = i;
    } else {
      sdCur = sd;
      sdItCur = i;
      if (sdCur - sdPre > 0) {
        sdFlagCur = INCREASE;
        ++continuousIncreaseCnt;
        continuousDecreaseCnt = 0;
      } else {
        sdFlagCur = DECREASE;
        continuousIncreaseCnt = 0;
        ++continuousDecreaseCnt;
      }
      if (continuousIncreaseCnt >= 3 || continuousDecreaseCnt >= 3) {
        break;
      }
      if ((sdFlagPre == DECREASE) && (sdFlagCur == INCREASE)) {
        break;
      } else {
        sdPre = sdCur;
        sdItPre = sdItCur;
        sdFlagPre = sdFlagCur;
      }
    }
    {
      IplImage* faceDownGrayRefine = cvCreateImage(cvGetSize(faceDwn), 8, 1);
      cvZero(faceDownGrayRefine);
      for (int y = 0; y < image_Lip->height; y++) {
        for (int x = 0; x < image_Lip->width; x++) {
          int pixel = cvGetReal2D(image_Lip, y, x);
          cvSetReal2D(faceDownGrayRefine, rect_Lip.y + y, rect_Lip.x + x,
                      pixel);
        }
      }
      nLipRefine = lpdt(faceDownGrayRefine);
      if (nLipRefine == 0) {
        if (sdItPre > 0) {
          sdItPre--;
        }
        sdItCur = (it + sdItPre) / 2;
        break;
      }
      cvReleaseImage(&faceDownGrayRefine);
    }
  }
  double T = 0.0;
  if (nLipRefine == 0) {
    T = (iterate_Threshold[sdItPre] +
         iterate_Threshold[sdItCur] /*+iterate_Threshold[it]*/) /
        2;
  } else {
    T = (iterate_Threshold[sdItPre] +
         iterate_Threshold[sdItPre] /*+iterate_Threshold[it]*/) /
        2;
  }
  for (int y = 0; y < image_LipBi->height; y++) {
    for (int x = 0; x < image_LipBi->width; x++) {
      if (cvGetReal2D(image_Gauss, y, x) >= T) {
        cvSetReal2D(image_LipBi, y, x, 0);
      }
    }
  }
  cvSmooth(image_LipBi, image_LipBi, CV_MEDIAN);
  ilmf(image_LipBi);
  cvReleaseImage(&faceDwn_Gray);
  cvReleaseImage(&image_Lip);
  cvReleaseImage(&image_Gauss);
  return flag_Lip;
}
void bbmm() {
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
  mComplexionGMM = new GMM(nGMM);
  mComplexionGMM->Build(data, nrows);
  for (int i = 0; i < nrows; i++) free(data[i]);
  free(data);
  cvReleaseImage(&faceImageScale);
  cvReleaseImage(&faceImageScaleGray);
  cvReleaseImage(&faceScaleEllipseMask);
  cvReleaseImage(&faceImageScale_Lab);
}
int copdt(const CvScalar skinColor) {
  int response = 5;
  float L;
  float feature[2];
  L = skinColor.val[0];
  feature[0] = skinColor.val[1];
  feature[1] = skinColor.val[2];
  if (L >= 165) {
    response = 0;
  } else if (L <= 90) {
    response = 1;
  } else {
    Mat testDataMat(1, 2, CV_32FC1, feature);
    //CvSVM svm = CvSVM();
//    CvSVM svm;
//    svm.load(colorClassifierPathName);
//    response = (int)svm.predict(testDataMat);

      Mat responses;
//        Ptr<cv::ml::SVM> svm = SVM::create();
      Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(colorClassifierPathName); //读取模型
      svm->predict(testDataMat, responses);
      responses.convertTo(responses,CV_32S);
      response = responses.at<int>(0,0);
  }
  return response;
}
typedef struct {
  double ScaleX;
  double ScaleY;
  double ScaleW;
  double ScaleH;
} deRctnglScale;
CvScalar glopt(IplImage* inputImage) {
  int size;
  size = (inputImage->height > inputImage->width) ? inputImage->height
                                                  : inputImage->width;
  double type = ((double)size) / 600.0;
  IplImage* faceImage = NULL;
  if (type <= 1) {
    faceImage = cvCloneImage(inputImage);
  } else {
    faceImage = cvCreateImage(cvSize(cvRound(inputImage->width / type),
                                     cvRound(inputImage->height / type)),
                              inputImage->depth, inputImage->nChannels);
    cvResize(inputImage, faceImage);
  }
  CvScalar glossResult;
  deRctnglScale cheekCoordScale = {0.65, 0.5, 0.6, 0.35};
  int imgScale = 1;
  int H = 100;
  int W = 100;
  CvRect cheekRect;
  cheekRect.x =
      cvRound((0 + faceImage->width * cheekCoordScale.ScaleX) * imgScale);
  cheekRect.y =
      cvRound((0 + faceImage->height * cheekCoordScale.ScaleY) * imgScale);
  cheekRect.width = cvRound(faceImage->width * (1 - cheekCoordScale.ScaleX) *
                            cheekCoordScale.ScaleW * imgScale);
  cheekRect.height = cvRound(faceImage->height * (1 - cheekCoordScale.ScaleY) *
                             cheekCoordScale.ScaleH * imgScale);
  cvSetImageROI(faceImage, cheekRect);
  IplImage* cheekImage = cvCreateImage(cvGetSize(faceImage), faceImage->depth,
                                       faceImage->nChannels);
  cvCopy(faceImage, cheekImage);
  cvResetImageROI(faceImage);
  Mat rszImageM;
  Mat cheekImageTemp;
  cheekImageTemp = cvarrToMat(cheekImage); //testj
  resize(cheekImageTemp, rszImageM, Size(W, H), 0, 0);
  Mat HSVImageM;
  cvtColor(rszImageM, HSVImageM, CV_RGB2HSV);
  Mat img32FC1;
  Mat SnglChnlImageM = Mat(H, 3 * W, CV_8UC1, (float*)HSVImageM.data);
  SnglChnlImageM.convertTo(img32FC1, CV_32FC1);
  Mat prjctdM;
  Mat mchIPM;
  Mat lssIPM;
  Mat nnIPM;
  FileStorage fs(glossModelPathName, FileStorage::READ);
  fs["prjctdM"] >> prjctdM;
  fs["mchIPM"] >> mchIPM;
  fs["lssIPM"] >> lssIPM;
  fs["nnIPM"] >> nnIPM;
  fs.release();
  Mat tstPatternM;
  gemm(img32FC1, prjctdM, 1, Mat(), 0, tstPatternM, GEMM_2_T);
  tstPatternM.cols = tstPatternM.cols * tstPatternM.rows;
  tstPatternM.rows = 1;

  Mat MDist(1, &mchIPM.rows, CV_64FC1);
  double tmp;
  for (int i = 0; i < mchIPM.rows; i++) {
    tmp = 1 - abs(tstPatternM.dot(mchIPM.row(i)) /
                  (norm(tstPatternM, NORM_L2) * norm(mchIPM.row(i), NORM_L2)));
    MDist.at<double>(i) = 1 - sqrt(1 - tmp * tmp);
  }
  minMaxLoc(MDist, 0, &glossResult.val[1]);
  MDist.release();

  Mat MDist1(1, &lssIPM.rows, CV_64FC1);
  for (int i = 0; i < lssIPM.rows; i++) {
    tmp = 1 - abs(tstPatternM.dot(lssIPM.row(i)) /
                  (norm(tstPatternM, NORM_L2) * norm(lssIPM.row(i), NORM_L2)));
    MDist1.at<double>(i) = 1 - sqrt(1 - tmp * tmp);
  }
  minMaxLoc(MDist1, 0, &glossResult.val[2]);
  MDist1.release();

  Mat MDist2(1, &nnIPM.rows, CV_64FC1);
  for (int i = 0; i < nnIPM.rows; i++) {
    tmp = 1 - abs(tstPatternM.dot(nnIPM.row(i)) /
                  (norm(tstPatternM, NORM_L2) * norm(nnIPM.row(i), NORM_L2)));
    MDist2.at<double>(i) = 1 - sqrt(1 - tmp * tmp);
  }
  minMaxLoc(MDist2, 0, &glossResult.val[3]);
  MDist2.release();
  glossResult.val[0] = glossResult.val[1] > glossResult.val[2]
                           ? (glossResult.val[1] > glossResult.val[3] ? 1 : 3)
                           : (glossResult.val[2] > glossResult.val[3] ? 2 : 3);
  glossResult.val[0] -= 1;
  cvReleaseImage(&faceImage);
  cvReleaseImage(&cheekImage);
  rszImageM.release();
  cheekImageTemp.release();
  HSVImageM.release();
  SnglChnlImageM.release();
  img32FC1.release();
  prjctdM.release();
  mchIPM.release();
  lssIPM.release();
  nnIPM.release();
  tstPatternM.release();
  return glossResult;
}
void lcet(const int maxStep) {
  IplImage* lipContourAnd = fipa(lipExtractMask, 5);
  fllc(lipContourAnd);
  for (int num = 10; num <= maxStep; num = num + 5) {
    IplImage* lipContour_num = fipa(lipExtractMask, num);
    fllc(lipContour_num);
    cvAnd(lipContourAnd, lipContour_num, lipContourAnd);
    cvReleaseImage(&lipContour_num);
  }
  ilmf(lipContourAnd);
  IplImage* lipContourOr = fipa(lipContourAnd, 5);
  fllc(lipContourOr);
  for (int num = 5; num <= maxStep; num = num + 5) {
    IplImage* lipContour_num = fipa(lipContourAnd, num);
    fllc(lipContour_num);
    cvOr(lipContourOr, lipContour_num, lipContourOr);
    cvReleaseImage(&lipContour_num);
  }
  cvReleaseImage(&lipExtractMask);
  lipExtractMask = cvCloneImage(lipContourOr);
  sglp();
  cvReleaseImage(&lipContourAnd);
  cvReleaseImage(&lipContourOr);
}
CvRect lple(IplImage* faceDown_Gray, bool& falg_Lip) {
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(faceDown_Gray, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  CvRect lipRect = cvRect(0, 0, 0, 0);
  int nlip = 0;
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);
    double ratio_WH = 0;
    double ratio_FILL = 0;
    double ratio_AREA = 0;
    CvPoint center;
    ratio_WH = ((double)(rect.width)) / ((double)(rect.height));
    ratio_FILL = ((double)area) / ((double)(rect.width * rect.height));
    ratio_AREA = ((double)area) /
                 ((double)(faceDown_Gray->width * faceDown_Gray->height));
    center.x = rect.x + rect.width / 2;
    center.y = rect.y + rect.height / 2;
    if ((ratio_WH > 1) && (ratio_WH <= 6) && (ratio_FILL >= 0.3) &&
        (ratio_FILL < 1) && (ratio_AREA >= 0.02) && (ratio_AREA <= 0.15) &&
        (center.x >= faceDown_Gray->width / 3) &&
        (center.x <= faceDown_Gray->width * 2 / 3) &&
        (center.y >= faceDown_Gray->height / 3)) {
      lipRect = rect;
      nlip++;
    }
  }
  if (nlip == 1) {
    falg_Lip = true;
  } else {
    falg_Lip = false;
  }
  cvReleaseMemStorage(&pStorage);
  return lipRect;
}
CvRect lpal(IplImage* faceDown_Gray, bool& falg_Lip, CvRect lip_region) {
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(faceDown_Gray, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  CvRect lipRect = cvRect(0, 0, 0, 0);
  int nlip = 0;
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);
    double ratio_WH = 0;
    double ratio_FILL = 0;
    double ratio_AREA = 0;
    CvPoint center;
    ratio_WH = ((double)(rect.width)) / ((double)(rect.height));
    ratio_FILL = ((double)area) / ((double)(rect.width * rect.height));
    ratio_AREA = ((double)area) /
                 ((double)(faceDown_Gray->width * faceDown_Gray->height));
    center.x = rect.x + rect.width / 2;
    center.y = rect.y + rect.height / 2;
    if ((ratio_WH > 1) && (ratio_WH <= 6) && (ratio_FILL >= 0.3) &&
        (ratio_FILL < 1) && (ratio_AREA >= 0.02) && (ratio_AREA <= 0.15) &&
        (center.x >= faceDown_Gray->width / 3) &&
        (center.x <= faceDown_Gray->width * 2 / 3) &&
        (center.y >= faceDown_Gray->height / 3) &&
        (center.y <= lip_region.y + lip_region.height) &&
        (center.y >= lip_region.y) &&
        (center.x <= lip_region.x + lip_region.width) &&
        (center.x >= lip_region.x)) {
      lipRect = rect;
      nlip++;
    }
  }
  if (nlip == 1) {
    falg_Lip = true;
  } else {
    falg_Lip = false;
  }
  cvReleaseMemStorage(&pStorage);
  return lipRect;
}
int llrf(IplImage* image_lip) {
  int s = 0;
  for (int y = 0; y < image_lip->height; y++) {
    for (int x = 0; x < image_lip->width; x++) {
      if (cvGetReal2D(image_lip, y, x) > 200) {
        s++;
      }
    }
  }
  return s;
}

int lpdt(IplImage* faceDown_Gray) {
  CvSeq* pContour = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(faceDown_Gray, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  int nlip = 0;
  for (; pContour != NULL; pContour = pContour->h_next) {
    int area = (int)cvContourArea(pContour);
    CvRect rect = cvBoundingRect(pContour);
    double ratio_WH = 0;
    double ratio_FILL = 0;
    double ratio_AREA = 0;
    CvPoint center;
    ratio_WH = ((double)(rect.width)) / ((double)(rect.height));
    ratio_FILL = ((double)area) / ((double)(rect.width * rect.height));
    ratio_AREA = ((double)area) /
                 ((double)(faceDown_Gray->width * faceDown_Gray->height));
    if ((ratio_WH > 1) && (ratio_WH <= 6) && (ratio_FILL >= 0.25) &&
        (ratio_FILL < 1) && (ratio_AREA >= 0.022) && (ratio_AREA <= 0.15)) {
      nlip++;
    }
  }
  cvReleaseMemStorage(&pStorage);
  return nlip;
}

void cadbt(const IplImage* mask, const IplImage* grayImage, int& i_dark,
           int& i_bright, double rate_dark, double rate_bright, float lowPart) {
  float data[256] = {0};
  int total = 0;
  for (int y = mask->height * lowPart; y < mask->height; y++) {
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

IplImage* gtlm() { return lipExtractMask; }

IplImage* gtli() { return lipExtractImage; }

void faii(IplImage* imageSrc1) {

  imageSrc = cvCloneImage(imageSrc1);
  if (storage) {
      cvReleaseMemStorage(&storage);
      storage = NULL;
  }
  cascade = NULL;
}

bool detect() {
  bool isFace = false;
  cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
  storage = cvCreateMemStorage(0);

  int scale = 1;

  int size;
  size =
      (imageSrc->height > imageSrc->width) ? imageSrc->height : imageSrc->width;
  scale = size / 600;
  if (scale <= 1) {
    scale = 1;
  }

  IplImage* gray =
      cvCreateImage(cvSize(imageSrc->width, imageSrc->height), 8, 1);
  IplImage* small_img = cvCreateImage(cvSize(cvRound(imageSrc->width / scale),
                                             cvRound(imageSrc->height / scale)),
                                      8, 1);
  cvCvtColor(imageSrc, gray, CV_BGR2GRAY);
  cvResize(gray, small_img, CV_INTER_LINEAR);
  cvEqualizeHist(small_img, small_img);
  cvClearMemStorage(storage);

  // cvShowImage("smallImage",small_img);

  CvRect* r;
  int cnt = 0;
  int rSize, maxSize = 0;
  if (cascade) {
    CvSeq* faces = cvHaarDetectObjects(small_img, cascade, storage, 1.1, 2,
                                       CV_HAAR_FIND_BIGGEST_OBJECT |
                                           CV_HAAR_DO_ROUGH_SEARCH |
                                           CV_HAAR_DO_CANNY_PRUNING,
                                       cvSize(100, 100));
    cnt = faces->total;
    for (int i = 0; i < (faces ? faces->total : 0); i++) {
      r = (CvRect*)cvGetSeqElem(faces, i);
      rSize = r->width * r->height;
      if (rSize > maxSize) {
        maxSize = rSize;
        faceRect = *r;
      }
    }
  } else {
    cout << "Loading haarcascade_frontalface_alt2.xml is wrong!!" << endl;
  }
  if (cnt == 0) {
    isFace = false;
  } else {
    isFace = true;
    faceRect.x = faceRect.x * scale;
    faceRect.y = faceRect.y * scale;
    faceRect.height = faceRect.height * scale;
    faceRect.width = faceRect.width * scale;
  }
  cvReleaseImage(&gray);
  gray = NULL;
  cvReleaseImage(&small_img);
  cvReleaseImage(&imageSrc);
  if (storage) {
      cvReleaseMemStorage(&storage);
      storage = NULL;
  }
  if (cascade) {
      cascade = NULL;
  }
  return isFace;
}

CvRect gfrt() { return faceRect; }

void ilmf(IplImage* mask) {
  IplImage* mask1 = cvCloneImage(mask);
  IplImage* mask2 = cvCloneImage(mask);

  CvSeq* pContour1 = NULL;
  CvMemStorage* pStorage1 = cvCreateMemStorage(0);
  int n1 = cvFindContours(mask1, pStorage1, &pContour1, sizeof(CvContour),
                          CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  int areaMax = 0;
  for (; pContour1 != NULL; pContour1 = pContour1->h_next) {
    int area = (int)cvContourArea(pContour1);
    if (area > areaMax) {
      areaMax = area;
    }
  }

  CvSeq* pContour2 = NULL;
  CvMemStorage* pStorage2 = cvCreateMemStorage(0);
  int n2 = cvFindContours(mask2, pStorage2, &pContour2, sizeof(CvContour),
                          CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  for (; pContour2 != NULL; pContour2 = pContour2->h_next) {
    int area = (int)cvContourArea(pContour2);
    CvRect rect = cvBoundingRect(pContour2);
    if (area < areaMax) {
      cvSetImageROI(mask, rect);
      cvSetZero(mask);
      cvResetImageROI(mask);
    }
  }

  cvReleaseMemStorage(&pStorage1);
  cvReleaseMemStorage(&pStorage2);
  cvReleaseImage(&mask1);
  cvReleaseImage(&mask2);
}

void sglp() {
  lipExtractImage = cvCloneImage(faceZ);
  for (int y = 0; y < lipExtractMask->height; y++) {
    for (int x = 0; x < lipExtractMask->width; x++) {
      if (cvGetReal2D(lipExtractMask, y, x) < 100) {
        cvSet2D(lipExtractImage, y, x, CV_RGB(0, 0, 0));
      }
    }
  }
}

void fsdmm(IplImage* src, const int haarFaceDetectSwitch /*=0*/) {
  inputImage = cvCloneImage(src);

  switchHaarFace = haarFaceDetectSwitch;
  isDetectFace = true;

  nGMM = 3;

  face = NULL;
  faceImage = NULL;
  faceImage_Lab = NULL;
  faceImage_Gauss = NULL;
  faceSkinImage = NULL;
  faceSkinMask = NULL;

  mComplexionGMM = NULL;
}

void dysal(const IplImage* inputImage) {
  int size;
  size = (inputImage->height > inputImage->width) ? inputImage->height
                                                  : inputImage->width;
  double type = ((double)size) / 300.0;

  if (type <= 1) {
    faceImage = cvCloneImage(inputImage);
  } else {
    faceImage = cvCreateImage(cvSize(cvRound(inputImage->width / type),
                                     cvRound(inputImage->height / type)),
                              inputImage->depth, inputImage->nChannels);
    cvResize(inputImage, faceImage);
  }
}

IplImage* fipa(IplImage* mask, int step) {
  IplImage* lipContour = cvCreateImage(cvGetSize(mask), 8, 1);
  cvZero(lipContour);

  CvPoint lipTopStartPoint = cvPoint(0, 0), lipTopEndPoint = cvPoint(0, 0);

  for (int x = 0; x < mask->width; x++) {
    bool flag = false;
    for (int y = 0; y < mask->height; y++) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        lipTopStartPoint.x = x;
        lipTopStartPoint.y = y;
        flag = true;
        break;
      }
    }
    if (flag) {
      break;
    }
  }
  for (int x = mask->width - 1; x >= 0; x--) {
    bool flag = false;
    for (int y = 0; y < mask->height; y++) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        lipTopEndPoint.x = x;
        lipTopEndPoint.y = y;
        flag = true;
        break;
      }
    }
    if (flag) {
      break;
    }
  }
  CvPoint point1 = lipTopStartPoint, point2 = cvPoint(0, 0);
  for (int x = lipTopStartPoint.x + step; x <= lipTopEndPoint.x; x = x + step) {
    for (int y = 0; y < mask->height; y++) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        point2.x = x;
        point2.y = y;
        break;
      }
    }
    cvLine(lipContour, point1, point2, cvScalarAll(255), 1, 8);
    point1 = point2;
  }
  cvLine(lipContour, point2, lipTopEndPoint, cvScalarAll(255), 1, 8);
  CvPoint lipDwnStartPoint = cvPoint(0, 0), lipDwnEndPoint = cvPoint(0, 0);

  for (int x = 0; x < mask->width; x++) {
    bool flag = false;
    for (int y = mask->height - 1; y >= 0; y--) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        lipDwnStartPoint.x = x;
        lipDwnStartPoint.y = y;
        flag = true;
        break;
      }
    }
    if (flag) {
      break;
    }
  }
  for (int x = mask->width - 1; x >= 0; x--) {
    bool flag = false;
    for (int y = mask->height - 1; y >= 0; y--) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        lipDwnEndPoint.x = x;
        lipDwnEndPoint.y = y;
        flag = true;
        break;
      }
    }
    if (flag) {
      break;
    }
  }
  point1 = lipDwnStartPoint;
  for (int x = lipDwnStartPoint.x + step; x <= lipDwnEndPoint.x; x = x + step) {
    for (int y = mask->height - 1; y >= 0; y--) {
      int pixel = cvGetReal2D(mask, y, x);
      if (pixel > 200) {
        point2.x = x;
        point2.y = y;
        break;
      }
    }
    cvLine(lipContour, point1, point2, cvScalarAll(255), 1, 8);
    point1 = point2;
  }
  cvLine(lipContour, point2, lipDwnEndPoint, cvScalarAll(255), 1, 8);

  cvLine(lipContour, lipTopStartPoint, lipDwnStartPoint, cvScalarAll(255), 1,
         8);
  cvLine(lipContour, lipTopEndPoint, lipDwnEndPoint, cvScalarAll(255), 1, 8);

  return lipContour;
}

void ccpp() {
  faceImage_Lab = cvCreateImage(cvGetSize(faceImage), 8, 3);
  cvCvtColor(faceImage, faceImage_Lab, CV_BGR2Lab);

  faceImage_Gauss = cvCreateImage(cvGetSize(faceImage), IPL_DEPTH_64F, 1);
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
}

void itds() {
  faceSkinMask = cvCreateImage(cvGetSize(faceImage), 8, 1);
  cvZero(faceSkinMask);

  IplImage* ellipseMask = cvCreateImage(cvGetSize(faceImage), 8, 1);
  cvZero(ellipseMask);
  CvPoint center = cvPoint(cvRound(ellipseMask->width * 0.5),
                           cvRound(ellipseMask->height * 0.5));
  CvSize size = cvSize(cvRound(ellipseMask->width * 0.36),
                       cvRound(ellipseMask->height * 0.50));
  cvEllipse(ellipseMask, center, size, 0, 0, 360, cvScalar(255), CV_FILLED);

  long ellipseMask_TotalCnt = 0;
  for (int y = 0; y < ellipseMask->height; ++y) {
    for (int x = 0; x < ellipseMask->width; ++x) {
      if (cvGetReal2D(ellipseMask, y, x) > 200) {
        ellipseMask_TotalCnt++;
      }
    }
  }

  double iterate_Threshold[17] = {0.3,   0.2,    0.1,    0.09,    0.07,   0.05,
                                  0.03,  0.01,   0.009,  0.007,   0.005,  0.003,
                                  0.001, 0.0005, 0.0001, 0.00005, 0.00001};

  long skinCntInEllipse = 0;
  for (int i = 0; i < 17; i++) {
    for (int y = 0; y < faceImage_Gauss->height; y++) {
      for (int x = 0; x < faceImage_Gauss->width; x++) {
        if (cvGetReal2D(faceImage_Gauss, y, x) >= iterate_Threshold[i]) {
          if (cvGetReal2D(faceSkinMask, y, x) < 100) {
            cvSetReal2D(faceSkinMask, y, x, 255);
            if (cvGetReal2D(ellipseMask, y, x) > 200) {
              skinCntInEllipse++;
            }
          }
        }
      }
    }

    double skinRate = (double)skinCntInEllipse / (double)ellipseMask_TotalCnt;

    if (skinRate > 0.82 || iterate_Threshold[i] < 0.01) {
      break;
    }
  }

  cvErode(faceSkinMask, faceSkinMask, NULL, 1);

  RemoveNoise rn;
  rn.LessConnectedRegionRemove(faceSkinMask,
                               faceSkinMask->height * faceSkinMask->width / 8);

  faceSkinImage = cvCloneImage(faceImage);
  for (int y = 0; y < faceSkinMask->height; ++y) {
    for (int x = 0; x < faceSkinMask->width; ++x) {
      if (cvGetReal2D(faceSkinMask, y, x) < 100) {
        cvSet2D(faceSkinImage, y, x, CV_RGB(255, 255, 255));
      }
    }
  }
  cvReleaseImage(&ellipseMask);
}

CvScalar efscf() {
  IplImage* faceImageGray = cvCreateImage(cvGetSize(faceImage), 8, 1);
  cvCvtColor(faceImage, faceImageGray, CV_BGR2GRAY);
  int i_dark, i_bright;
  double rate_dark = 0.1, rate_bright = 0.02;
  cdbt(faceSkinMask, faceImageGray, i_dark, i_bright, rate_dark, rate_bright);
  for (int y = 0; y < faceSkinMask->height; ++y) {
    for (int x = 0; x < faceSkinMask->width; ++x) {
      if (cvGetReal2D(faceImageGray, y, x) < i_dark ||
          cvGetReal2D(faceImageGray, y, x) > i_bright) {
        cvSetReal2D(faceSkinMask, y, x, 0);
      }
    }
  }

  CvScalar faceSkinColorFeature = cvAvg(faceImage_Lab, faceSkinMask);

  cvReleaseImage(&faceImageGray);

  return faceSkinColorFeature;
}

void fllc(IplImage* contour) {
  IplImage* contourCopy = cvCloneImage(contour);
  CvSeq* pContour = NULL;
  CvSeq* pConInner = NULL;
  CvMemStorage* pStorage = cvCreateMemStorage(0);
  int n = cvFindContours(contourCopy, pStorage, &pContour, sizeof(CvContour),
                         CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  cvDrawContours(contour, pContour, CV_RGB(255, 255, 255),
                 CV_RGB(255, 255, 255), 2, CV_FILLED, 8);

  for (; pContour != NULL; pContour = pContour->h_next) {
    for (pConInner = pContour->v_next; pConInner != NULL;
         pConInner = pConInner->h_next) {
      cvDrawContours(contour, pConInner, CV_RGB(255, 255, 255),
                     CV_RGB(255, 255, 255), 0, CV_FILLED, 8, cvPoint(0, 0));
    }
  }
  cvReleaseMemStorage(&pStorage);
  cvReleaseImage(&contourCopy);
}

bool fsdbmpf() {
  CvRect faceROIRect;
  if (switchHaarFace == 1) {
    faii(inputImage);
    // FaceDetection faceDetect = FaceDetection(inputImage);
    // isDetectFace = faceDetect.detect();
    isDetectFace = detect();
    if (!isDetectFace) {
      return isDetectFace;
    } else {
      faceROIRect = gfrt();
      faceROIRect.y = faceROIRect.y - faceROIRect.height / 10;
      faceROIRect.height = faceROIRect.height * 6 / 5;
      if (faceROIRect.y < 0) faceROIRect.y = 0;
      if (faceROIRect.height > inputImage->height)
        faceROIRect.height = inputImage->height;
    }
  } else {
    faceROIRect.x = 0;
    faceROIRect.y = 0;
    faceROIRect.width = inputImage->width;
    faceROIRect.height = inputImage->height;
  }

  cvSetImageROI(inputImage, faceROIRect);
  face = cvCreateImage(cvGetSize(inputImage), inputImage->depth,
                       inputImage->nChannels);
  cvCopy(inputImage, face);
  cvResetImageROI(inputImage);

  dysal(face);

  bbmm();
  ccpp();
  itds();

  return isDetectFace;
}

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

IplImage* gfsi() { return faceSkinImage; }

IplImage* gefit() { return face; }
