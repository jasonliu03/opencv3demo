#include <jni.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <iostream>


#include "GMM_face.h"
#include "GMM.h"
#include "faceDignosis.h"
#include "tongueDiagnosis.h"
#include "imagehandle.h"
#include "TCMProInterface.h"
#include "RemoveNoise.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_zhiyuntcm_opencv3demo_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


//extern "C" JNIEXPORT jintArray
//JNICALL
//Java_com_zhiyuntcm_opencv3demo_MainActivity_animationFaceMask
//        (
//                JNIEnv *env,
//                jobject /* this */,jintArray buf,jint w,jint h) {
//    jint *cbuf;
//    jboolean ptfalse = false;
//    cbuf = env->GetIntArrayElements(buf, &ptfalse);
//    if(cbuf == NULL){
//        return 0;
//    }
//
//    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
//    // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
//    IplImage iplImgData;
//    iplImgData= (IplImage)imgData;
//    IplImage *input = &iplImgData;
//    GMM *mGMM = bbmm(input);
//    IplImage *faceImage_Gauss = ccpp(input, mGMM);
//    Mat rstFace;
//    rstFace = cvarrToMat(faceImage_Gauss);
//
//    Mat detect(imgData.rows, imgData.cols, CV_8UC4, Scalar(0, 0, 0));
//    Mat srcWhite(imgData.rows, imgData.cols, CV_8UC4, Scalar(100, 100, 255));
//    srcWhite.copyTo(detect, rstFace);
//    int size = w * h;
//    jintArray result = env->NewIntArray(size);
//    env->SetIntArrayRegion(result, 0, size, (jint*)detect.data);
//    env->ReleaseIntArrayElements(buf, cbuf, 0);
//
//    return result;
//}

extern "C" JNIEXPORT jintArray
JNICALL
Java_com_zhiyuntcm_opencv3demo_MainActivity_animationTongueSegmentation
        (
                JNIEnv *env,
                jobject /* this */,jintArray buf,jint w,jint h) {
    jint *cbuf;
    jboolean ptfalse = false;
    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    if(cbuf == NULL){
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
    cvtColor( imgData, imgData, CV_BGRA2BGR );
    // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
    Mat dstMat = tongueDivision(imgData);
//    cvtColor( dstMat, dstMat, CV_BGR2BGRA );
    cvtColor(dstMat, dstMat, CV_GRAY2BGR);
    cvtColor(dstMat, dstMat, CV_BGR2BGRA);
    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint*)dstMat.data);
    env->ReleaseIntArrayElements(buf, cbuf, 0);

    return result;
}

extern "C" JNIEXPORT jintArray
JNICALL
Java_com_zhiyuntcm_opencv3demo_MainActivity_animationTongueMask
        (
                JNIEnv *env,
                jobject /* this */,jintArray buf,jint w,jint h) {
    jint *cbuf;
    jboolean ptfalse = false;
    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    if(cbuf == NULL){
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
    // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
    IplImage iplImgData;
    iplImgData= (IplImage)imgData;
    IplImage *input = &iplImgData;
    GMM *mGMM = bbmm(input);
    IplImage *faceImage_Gauss = ccpp(input, mGMM);
    cvThreshold(faceImage_Gauss,faceImage_Gauss,0,255,CV_THRESH_BINARY);

    cvMorphologyEx(faceImage_Gauss, faceImage_Gauss, NULL, NULL,CV_MOP_OPEN,4);

    RemoveNoise rn;
    rn.LessConnectedRegionRemove(faceImage_Gauss,
                                 faceImage_Gauss->height * faceImage_Gauss->width / 8);
    Mat rstFace;
    rstFace = cvarrToMat(faceImage_Gauss);

    cvtColor(rstFace, rstFace, CV_GRAY2BGRA);

//    Mat detect(imgData.rows, imgData.cols, CV_8UC4, Scalar(0, 0, 0));
//    Mat srcWhite(imgData.rows, imgData.cols, CV_8UC4, Scalar(100, 100, 255));
//    srcWhite.copyTo(detect, rstFace);
    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint*)rstFace.data);
    env->ReleaseIntArrayElements(buf, cbuf, 0);

    return result;
}

extern "C" JNIEXPORT jstring
JNICALL
Java_com_zhiyuntcm_opencv3demo_MainActivity_tcmFacePro
        (
                JNIEnv *env,
                jobject /* this */,jintArray buf,jint w,jint h,jstring cascadeFileName, jstring svmFace, jstring glossFace, jstring cascadeLip, jstring svmLip) {
    const char* pstrCascadeFileName = env->GetStringUTFChars(cascadeFileName, NULL);
    cascade_name = pstrCascadeFileName;
    const char* pstrSvmFace = env->GetStringUTFChars(svmFace, NULL);
    colorClassifierPathName = pstrSvmFace;
    const char* pstrGlossFace = env->GetStringUTFChars(glossFace, NULL);
    glossModelPathName = pstrGlossFace;
    const char* pstrCascadeLip = env->GetStringUTFChars(cascadeLip, NULL);
    cascade_nameL = pstrCascadeLip;
    const char* pstrSvmLip = env->GetStringUTFChars(svmLip, NULL);
    colorClassifierPathNameZ = pstrSvmLip;
    jint *cbuf;
    jboolean ptfalse = false;
    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    if(cbuf == NULL){
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
    cvtColor( imgData, imgData, CV_BGRA2BGR );
    // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
    IplImage iplImgData;
    iplImgData= (IplImage)imgData;
    IplImage *input = &iplImgData;
    char* rstFace = tcmFacePro(input, 1);

    return env -> NewStringUTF(rstFace);
}

extern "C" JNIEXPORT jstring
JNICALL
Java_com_zhiyuntcm_opencv3demo_MainActivity_tcmTonguePro
        (
                JNIEnv *env,
                jobject /* this */,jintArray buf,jint w,jint h,jstring cascadeFileName, jstring jccolorClassifierPathName, jstring jbohouClassifierPathName, jstring jpanClassifierPathName) {
    const char* pstrCascadeFileName = env->GetStringUTFChars(cascadeFileName, NULL);
    cascade_nameS = pstrCascadeFileName;
    const char* pstrccolorClassifierPathName = env->GetStringUTFChars(jccolorClassifierPathName, NULL);
    ccolorClassifierPathName = pstrccolorClassifierPathName;
    const char* pstrbohouClassifierPathName = env->GetStringUTFChars(jbohouClassifierPathName, NULL);
    bohouClassifierPathName = pstrbohouClassifierPathName;
    const char* pstrpanClassifierPathName = env->GetStringUTFChars(jpanClassifierPathName, NULL);
    panClassifierPathName = pstrpanClassifierPathName;
    jint *cbuf;
    jboolean ptfalse = false;
    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    if(cbuf == NULL){
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);
    cvtColor( imgData, imgData, CV_BGRA2BGR );
    // 注意，Android的Bitmap是ARGB四通道,而不是RGB三通道
    IplImage iplImgData;
    iplImgData= (IplImage)imgData;
    IplImage *input = &iplImgData;
    char* rstTongue = tcmTonguePro(input, 1);

    return env -> NewStringUTF(rstTongue);
}
