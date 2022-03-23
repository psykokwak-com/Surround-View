#pragma once

#include "TCVideoController.h"

class TCVideoClientTimeoutHandler
{
public:
  TCVideoClientTimeoutHandler();
  void reset();
  bool isTimeout();
  int getTimeout(); // in ms

  static int checkInterrupt(void *ctx);

private:
  uint32_t _timeout;
  uint32_t _last;
};

class TCVideoClient : public TCVideoController
{
public:
  AVPixelFormat _hwPixFmt;

protected:
  std::thread  *_pVideoThread;
  std::mutex _mutexVideoThread;
  bool _videoThreadRunning;
  std::string _url;
  TCSDLTexture _texture;
  TCVideoClientTimeoutHandler _videoTimeoutHandler;
  std::string _name;
  int _rtspTimeout;
  std::string _transport;

private:

public:
  TCVideoClient(const std::string &name);
  virtual ~TCVideoClient();

  void setUrl(const std::string &url);
  TCVideoClientTimeoutHandler *getTimeoutHandler();
  int worker();
  int workerStream();

  void setOverrideRTSPTimeout(int t); // in seconds
  void setTransport(const std::string &transport);
  virtual int play();
  virtual int stop();
  bool running();
  bool playing();

  bool readLastFrame(cv::cuda::GpuMat &gpuFrame);

  int getInfoResolutionX();
  int getInfoResolutionY();
  std::string getInfoName();
};

