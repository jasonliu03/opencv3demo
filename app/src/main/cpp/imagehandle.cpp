#include"imagehandle.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc/types_c.h"

int DetectThreshold(IplImage*src)
{
	uchar iThrehold;
	try
	{
		int height = src->height;
		int width = src->width;
		int step = src->widthStep / sizeof(uchar);
		uchar *data = (uchar*)src->imageData;

		int iDiffRec = 0;
		int F[256] = { 0 };  //ֱ��ͼ����  
		int iTotalGray = 0;  //�Ҷ�ֵ��  
		int iTotalPixel = 0; //��������  
		uchar bt;			 //ĳ�������ֵ  

		uchar iNewThrehold;
		uchar iMaxGrayValue = 0, iMinGrayValue = 255;
		uchar iMeanGrayValue1, iMeanGrayValue2;

		//��ȡ(i,j)��ֵ������ֱ��ͼ����F  
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				bt = data[i*step + j];
				if (bt < iMinGrayValue)
					iMinGrayValue = bt;
				if (bt > iMaxGrayValue)
					iMaxGrayValue = bt;
				F[bt]++;
			}
		}

		iThrehold = 0;
		iNewThrehold = (iMinGrayValue + iMaxGrayValue) / 2;
		iDiffRec = iMaxGrayValue - iMinGrayValue;

		for (int a = 0; (abs(iThrehold - iNewThrehold) > 0.5); a++)
		{
			iThrehold = iNewThrehold;
			//С�ڵ�ǰ��ֵ���ֵ�ƽ���Ҷ�ֵ  
			for (int i = iMinGrayValue; i < iThrehold; i++)
			{
				iTotalGray += F[i] * i;
				iTotalPixel += F[i];
			}
			iMeanGrayValue1 = (uchar)(iTotalGray / iTotalPixel);
			//���ڵ�ǰ��ֵ���ֵ�ƽ���Ҷ�ֵ  
			iTotalPixel = 0;
			iTotalGray = 0;
			for (int j = iThrehold + 1; j < iMaxGrayValue; j++)
			{
				iTotalGray += F[j] * j;
				iTotalPixel += F[j];
			}
			iMeanGrayValue2 = (uchar)(iTotalGray / iTotalPixel);

			iNewThrehold = (iMeanGrayValue2 + iMeanGrayValue1) / 2;
			iDiffRec = abs(iMeanGrayValue2 - iMeanGrayValue1);
		}
	}
	catch (cv::Exception e)
	{
	}

	return iThrehold;
}

Mat Binary(Mat imgInput) {
	Mat imgGray, result;
	cvtColor(imgInput, imgGray, CV_BGR2GRAY);
	blur(imgGray, imgGray, Size(3, 3));
	IplImage pBinary = (IplImage)imgInput;
	IplImage *binary = &pBinary;
	IplImage *input = cvCloneImage(binary);
	int detectThreshold = DetectThreshold(input);
	threshold(imgGray, result, detectThreshold, 255, CV_THRESH_BINARY);
	cvReleaseImage(&input);
	return result;
}

Mat sobel(Mat inputImage) {
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;

	Sobel(inputImage, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(inputImage, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	return dst;
}

Mat RGB2HSV(Mat inputImage) {
	Mat HSVImage;
	cvtColor(inputImage, HSVImage, CV_RGB2HSV);
	for (int i = 0; i <= HSVImage.rows - 1; i++) {
		for (int j = 0; j <= HSVImage.cols - 1; j++) {
			HSVImage.at<Vec3b>(i, j)[1] = 255;
		}
	}
	Mat RGBImage;
	cvtColor(HSVImage, RGBImage, CV_HSV2RGB);
	HSVImage.release();
	return RGBImage;
}

Mat RGB2Lab(Mat inputImage) {
  	Mat LabImage;
	cvtColor(inputImage, LabImage, CV_RGB2Lab);
    return LabImage;
}

IplImage* DetectRelief(IplImage* input) {
	IplImage *detectRImage = cvCloneImage(input);
	int width = detectRImage->width;
	int height = detectRImage->height;
	int step = detectRImage->widthStep;
	int channel = detectRImage->nChannels;
	uchar* data = (uchar *)detectRImage->imageData;
	for (int i = 0;i<width - 1;i++)
	{
		for (int j = 0;j<height - 1;j++)
		{
			for (int k = 0;k<channel;k++)
			{
				int temp = data[(j + 1)*step + (i + 1)*channel + k] - data[j*step + i*channel + k] + 128;

				if (temp>255)
				{
					data[j*step + i*channel + k] = 255;
				}
				else if (temp<0)
				{
					data[j*step + i*channel + k] = 0;
				}
				else
				{
					data[j*step + i*channel + k] = temp;
				}
			}
		}
	}
	return detectRImage;
}

Mat Relief(Mat imgInput) {
	Mat result;
	IplImage pRelief = (IplImage)imgInput;
	IplImage* reliefImage = &pRelief;
	IplImage* input = cvCloneImage(reliefImage);
	IplImage* detectR = DetectRelief(input);
	Mat imageRelief;
	imageRelief = cvarrToMat(detectR);
	cvReleaseImage(&input);
	return imageRelief;
}

Mat sketch(Mat imgInput)
{
	int width = imgInput.cols;
	int heigh = imgInput.rows;
	Mat gray0, gray1;
	cvtColor(imgInput, gray0, CV_BGR2GRAY);
	addWeighted(gray0, -1, NULL, 0, 255, gray1);
	GaussianBlur(gray1, gray1, Size(11, 11), 0);
	Mat img(gray1.size(), CV_8UC1);
	for (int y = 0; y<heigh; y++)
	{
		uchar* P0 = gray0.ptr<uchar>(y);
		uchar* P1 = gray1.ptr<uchar>(y);
		uchar* P = img.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			int tmp0 = P0[x];
			int tmp1 = P1[x];
			P[x] = (uchar)min((tmp0 + (tmp0*tmp1) / (256 - tmp1)), 255);
		}
	}
	return img;
}

Mat sumiao(Mat Image_in)
{

	Mat Image_out(Image_in.size(), CV_32FC3);
	Image_in.convertTo(Image_out, CV_32FC3);

	Mat I(Image_in.size(), CV_32FC1);

	cv::cvtColor(Image_out, I, CV_BGR2GRAY);
	I = I / 255.0;

	Mat I_invert;
	I_invert = -I + 1.0;

	Mat I_gau;
	GaussianBlur(I_invert, I_gau, Size(25, 25), 0, 0);

	float delta = 0.01;
	I_gau = -I_gau + 1.0 + delta;

	Mat I_dst;
	cv::divide(I, I_gau, I_dst);
	I_dst = I_dst;

	Mat b(Image_in.size(), CV_32FC1);
	Mat g(Image_in.size(), CV_32FC1);
	Mat r(Image_in.size(), CV_32FC1);

	Mat rgb[] = { b,g,r };

	float alpha = 0.75;

	r = alpha*I_dst + (1 - alpha)*200.0 / 255.0;
	g = alpha*I_dst + (1 - alpha)*205.0 / 255.0;
	b = alpha*I_dst + (1 - alpha)*105.0 / 255.0;

	cv::merge(rgb, 3, Image_out);
	return Image_out;
}

Mat tongueDivision(Mat tongueROI) {

	IplImage ptongueROI = (IplImage)tongueROI;

	IplImage* tongueROIImage = &ptongueROI;

	IplImage* image_tongueROI = cvCloneImage(tongueROIImage);

	double roiScale = 1.0;

	int M = 45;

	IplImage* tmpScaleTongueROI = cvCreateImage(

		cvSize(cvGetSize(image_tongueROI).width*roiScale, cvGetSize(image_tongueROI).height*roiScale), IPL_DEPTH_8U, 3);
//    Mat tmpMatScaleTongueROI(tongueROI.size(), CV_8UC4, Scalar(0,0,0));
//
//    IplImage tmpIplScaleTongueROI = (IplImage)tmpMatScaleTongueROI;
//    IplImage* tmpScaleTongueROI = &tmpIplScaleTongueROI;
//
	cvResize(image_tongueROI, tmpScaleTongueROI);



	IplImage* image_vPolar = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	IplImage* image_v = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	IplImage* image_vBiPolar = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	cvLogPolar(image_v, image_vPolar,

		cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,

		CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);



	IplImage* image_vBiPolarcftt = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	IplImage* image_vMask = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	IplImage* image_vPolarEdge = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);



	cvSmooth(tmpScaleTongueROI, tmpScaleTongueROI, CV_GAUSSIAN);

	IplImage* image_hsv = cvCreateImage(

		cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 3);

	IplImage* image_h = cvCreateImage(

		cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 1);

	IplImage* image_s = cvCreateImage(

		cvSize(tmpScaleTongueROI->width, tmpScaleTongueROI->height), IPL_DEPTH_8U, 1);

	cvCvtColor(tmpScaleTongueROI, image_hsv, CV_BGR2HSV);

	cvSplit(image_hsv, image_h, image_s, image_v, NULL);



	for (int y = 0; y < image_vPolar->height; y++) {

		for (int x = 0; x < image_vPolar->width; x++) {

			if ((x < 6) || (x > image_vPolar->width - 7)) {

				cvSetReal2D(image_vPolarEdge, y, x, 0);

			}

			else {

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



	cvSmooth(image_vPolarEdgeInverseOSTU, image_vPolarEdgeInverseOSTU, CV_GAUSSIAN);



	IplImage* image_vClose = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_OPEN, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_CLOSE, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_CLOSE, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_OPEN, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_CLOSE, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_CLOSE, 4);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_ERODE, 2);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL, CV_MOP_DILATE, 3);

	cvMorphologyEx(image_vPolarEdgeInverseOSTU, image_vClose, NULL, NULL,CV_MOP_DILATE,3);



	cvLogPolar(image_vClose, image_vBiPolar,

		cvPoint2D32f(image_v->width / 2, image_v->height / 2), M,

		CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);

	cvThreshold(image_vBiPolar, image_vBiPolar, 0, 255,

		CV_THRESH_BINARY | CV_THRESH_OTSU);  //�ɿ��ǲ��ô����ֵ







	IplImage* image_vBiPolarFill = cvCreateImage(

		cvSize(image_tongueROI->width, image_tongueROI->height), IPL_DEPTH_8U, 1);

	image_vBiPolarFill = cvCloneImage(image_vBiPolarcftt);

	for (int y = 0; y < image_vBiPolarFill->height - 1; y++) {

		for (int x = 0; x < image_vBiPolarFill->width - 1; x++) {

			if (cvGetReal2D(image_vBiPolarFill, y, x) < 100) {

				cvSetReal2D(image_vBiPolarFill, y, x, 255);

			}

			else {

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



	for (int y = 0; y < image_vMask->height; y++) {

		for (int x = 0; x < image_vMask->width; x++) {

			if (cvGetReal2D(image_vMask, y, x) < 100) {

				cvSet2D(image_tongueROI, y, x, cvScalar(255, 255, 255));     // �ָ���ͷ��ɫ��Ϊ��ɫ��255��255��255������֮ǰΪ��ɫ��0��0��0��

			}

		}

	}

	Mat tongueROIImg;

	tongueROIImg = cvarrToMat(image_tongueROI);



	cvReleaseImage(&tmpScaleTongueROI);

	cvReleaseImage(&image_vBiPolar);

	cvReleaseImage(&image_vBiPolarcftt);

	cvReleaseImage(&image_vMask);

	cvReleaseImage(&image_vPolarEdge);

	cvReleaseImage(&image_hsv);

	cvReleaseImage(&image_h);

	cvReleaseImage(&image_s);

	cvReleaseImage(&image_v);

	cvReleaseImage(&image_vPolar);

	cvReleaseImage(&image_vPolarEdgeInverse);

	cvReleaseImage(&image_vPolarEdgeInverseOSTU);

	cvReleaseImage(&image_vClose);

	cvReleaseImage(&image_vBiPolarFill);

	cvReleaseImage(&image_vOSTU);

	return tongueROIImg;

}
