#pragma once
#include <iostream>
#include <string>


#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>

#include <cuda_runtime.h>

//#include <linux/videodev2.h>

using namespace std::literals::chrono_literals;


using uchar = unsigned char;

/*
    Video4linux 2
*/

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720
#define CAM_NUMS 4


typedef struct 
{
  cv::Size resolution;
  std::array<double, 9> K;
  std::array<double, 14> distortion;
  cv::Size captureResolution;
  bool read(const std::string& filepath, const int camNum, const cv::Size& resol = cv::Size(1920, 1080), const cv::Size& cameraResol = cv::Size(1920, 1080));
private:
  int _cameraNum;
} InternalCameraParams;

typedef struct {
  cv::cuda::GpuMat remapX, remapY;
  cv::cuda::GpuMat undistFrame;
  cv::Rect roiFrame;
} CameraUndistortData;

typedef struct {
  cv::cuda::GpuMat gpuFrame;
} Frame;

class MultiCameraSource
{
private:
  cv::VideoCapture *_pVideoCapture[CAM_NUMS];
  cv::Size _size;
  std::array<InternalCameraParams, CAM_NUMS> _camParams;
  std::array<CameraUndistortData, CAM_NUMS> _camUndistort;
  uchar* _cudaManagedMemory;

public:
  MultiCameraSource();
  ~MultiCameraSource();

  int init(const std::string& param_filepath, const cv::Size& calibSize, const cv::Size& undistSize, const bool useUndist = false);

  bool startStream();
  bool stopStream();
  void setFrameSize(const cv::Size& size);
  bool capture(std::array<Frame, CAM_NUMS>& frames);
};


















