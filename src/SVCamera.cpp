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

//#define DUMMY_STREAM


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
  cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::SHARED));

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
#ifdef DUMMY_STREAM
  return true;
#endif

  std::vector<std::string> videoCaptureUrl;


  videoCaptureUrl.push_back("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=1&audio=0&pull=1"); // Front
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=1&audio=0&pull=1");
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=2&audio=0&pull=1"); // Back
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.50.3/axis-media/media.amp?camera=2&audio=0&pull=1");

  /*
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.255.40/axis-media/media.amp?camera=1&audio=0&pull=1");
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.255.40/axis-media/media.amp?camera=1&audio=0&pull=1");
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.255.40/axis-media/media.amp?camera=1&audio=0&pull=1");
  videoCaptureUrl.push_back("rtsp://root:axis@192.168.255.40/axis-media/media.amp?camera=1&audio=0&pull=1");
  */

#ifndef NO_OMP
#pragma omp parallel for default(shared)
#endif
  for (int i = 0; i < CAM_NUMS; i++) {
    if (_pVideoCapture[i])
      continue;

    _pVideoCapture[i] = new TCVideoClient("camera " + std::to_string(i));
    _pVideoCapture[i]->setUrl(videoCaptureUrl[i]);
    _pVideoCapture[i]->setOverrideRTSPTimeout(5);
    _pVideoCapture[i]->setTransport("udp");
    _pVideoCapture[i]->play();

    LOG_DEBUG("Opening camera - %d ... OK", i);
  }

  do {
    bool ok = true;

    for (int i = 0; i < CAM_NUMS; i++)
      if (!_pVideoCapture[i]->playing()) ok = false;

    if (ok)
      break;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));;

  } while (42);

  return true;
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
#pragma omp parallel for default(none) shared(frames)
#endif
  for (int i = 0; i < CAM_NUMS; i++) {


#ifdef DUMMY_STREAM
    cv::cuda::GpuMat gpuFrame(_size, CV_8UC3, 0xFFFFFF);
#else
    cv::cuda::GpuMat gpuFrame(_size, CV_8UC3);
    if (!_pVideoCapture[i]->readLastFrame(gpuFrame))
      continue;
#endif

    cv::cuda::remap(gpuFrame, _camUndistort[i].undistFrame, _camUndistort[i].remapX, _camUndistort[i].remapY, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar());
    frames[i].gpuFrame = _camUndistort[i].undistFrame(_camUndistort[i].roiFrame);
}
  return true;
}