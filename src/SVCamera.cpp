#include <ctime>
#include <fcntl.h>
//#include <sys/ioctl.h>
//#include <unistd.h>

#include <thread>
#include <SVCamera.hpp>

#include <cusrc/yuv2rgb.cuh>

#include <chrono>

#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


#include <omp.h>


#define LOG_DEBUG(msg, ...)   printf("DEBUG:   " msg "\n", ##__VA_ARGS__)
#define LOG_WARNING(msg, ...) printf("WARNING: " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...)   printf("ERROR:   " msg "\n", ##__VA_ARGS__)



bool InternalCameraParams::read(const std::string& filepath, const int num, const cv::Size& resol, const cv::Size& camResol)
{
  std::string fnk = filepath + std::to_string(num) + ".K";
  std::string fnd = filepath + std::to_string(num) + ".dist";
  std::ifstream ifstrK{ fnk };
  std::ifstream ifstrDist{ fnd };


  if (!ifstrK.is_open() || !ifstrDist.is_open()) {
    LOG_ERROR("Can't opened file with internal camera params");
    return false;
  }

  for (size_t i = 0; i < 9; i++)
    ifstrK >> K[i];
  for (size_t j = 0; j < 14; ++j)
    ifstrDist >> distortion[j];

  captureResolution = camResol;
  resolution = resol;
  ifstrK.close();
  ifstrDist.close();

  return true;
}


// ----------------------END SECTION------------------------


MultiCameraSource::MultiCameraSource()
{
  memset(_pVideoCapture, 0, sizeof(_pVideoCapture));

  cudaMallocManaged(&_cudaManagedMemory, CAMERA_WIDTH * CAMERA_HEIGHT * 4 * 4, cudaMemAttachGlobal);

  cudaDeviceSynchronize();
}

MultiCameraSource::~MultiCameraSource()
{
  stopStream();

  cudaFree(_cudaManagedMemory);
}


bool MultiCameraSource::startStream()
{
  for (int i = 0; i < CAM_NUMS; i++) {
    if (_pVideoCapture[i])
      continue;

    _pVideoCapture[i] = new cv::VideoCapture();
    _pVideoCapture[i]->set(cv::CAP_PROP_BUFFERSIZE, 1);
    _pVideoCapture[i]->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000);
    _pVideoCapture[i]->set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000);
  }

  _pVideoCapture[0]->open("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=2&audio=0&pull=1"); // Back
  _pVideoCapture[1]->open("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=4&audio=0&pull=1");
  _pVideoCapture[2]->open("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=1&audio=0&pull=1"); // Front
  _pVideoCapture[3]->open("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=4&audio=0&pull=1");

  for (int i = 0; i < CAM_NUMS; i++)
    LOG_DEBUG("Opening camera - %d ... %s", i, _pVideoCapture[1]->isOpened() ? "OK :)" : "NOK :(");


  if (_pVideoCapture[0]->isOpened() && _pVideoCapture[1]->isOpened() && _pVideoCapture[2]->isOpened() && _pVideoCapture[3]->isOpened())
    return true;

  stopStream();

  return false;
}

bool MultiCameraSource::stopStream()
{
  for (int i = 0; i < CAM_NUMS; i++)
    if (_pVideoCapture[i]) {
      delete _pVideoCapture[i];
      _pVideoCapture[i] = NULL;
    }

  return true;
}

void MultiCameraSource::setFrameSize(const cv::Size& size)
{
  _size = size;
}

int MultiCameraSource::init(const std::string& param_filepath, const cv::Size& calibSize, const cv::Size& undistSize, const bool useUndist /*= false*/)
{
  for (int i = 0; i < CAM_NUMS; ++i)
  {
    if (param_filepath.empty()) {
      LOG_ERROR("Invalid input path with parameter...");
      return -1;
    }

    _camParams[i].read(param_filepath, i, calibSize, _size);

    cv::Mat K(3, 3, CV_64FC1);
    for (size_t k = 0; k < _camParams[i].K.size(); ++k)
      K.at<double>(k) = _camParams[i].K[k];

    cv::Mat D(_camParams[i].distortion);
    const cv::Size calibratedFrameSize(_camParams[i].resolution);
    auto& uData = _camUndistort[i];
    cv::Mat newK;

    if (useUndist)
      newK = cv::getOptimalNewCameraMatrix(K, D, undistSize, 1, undistSize, &uData.roiFrame); // 0.0 ? 1.0
    else
      newK = cv::getOptimalNewCameraMatrix(K, D, calibratedFrameSize, 1, undistSize, &uData.roiFrame); // 0.0 ? 1.0

    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(K, D, cv::Mat(), newK, undistSize, CV_32FC1, mapX, mapY);
    uData.remapX.upload(mapX);
    uData.remapY.upload(mapY);

    LOG_DEBUG("Generating undistort maps for camera - %d ... OK", i);
  }

  return 0; // OK
}

bool MultiCameraSource::capture(std::array<Frame, CAM_NUMS>& frames)
{
#ifndef NO_OMP
//pragma omp parallel for default(none)
#endif
  for (int i = 0; i < CAM_NUMS; i++) {
    cv::Mat cpuFrame;// (_size, CV_8UC3);
    cv::cuda::GpuMat gpuFrame;

    if (!_pVideoCapture[i]->read(cpuFrame))
      return false;

    gpuFrame.upload(cpuFrame);

    cv::cuda::remap(gpuFrame, _camUndistort[i].undistFrame, _camUndistort[i].remapX, _camUndistort[i].remapY, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar());
    frames[i].gpuFrame = _camUndistort[i].undistFrame(_camUndistort[i].roiFrame);
  }
  return true;
}