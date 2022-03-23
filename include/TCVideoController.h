#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <cmath>
#include <chrono>
#include <ctime>
#include <queue>
#include <algorithm>
#include <thread>

#include <chrono>

extern "C"
{
  // Build from https://www.gyan.dev/ffmpeg/builds/
#include <libavutil/time.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>

#ifdef _DEBUG
#define SDL_Log(msg, ...)   printf("DEBUG:   " msg "\n", ##__VA_ARGS__)
#define SDL_LogWarn(a, msg, ...) printf("WARNING: " msg "\n", ##__VA_ARGS__)
#define SDL_LogError(a, msg, ...)   printf("ERROR:   " msg "\n", ##__VA_ARGS__)
#else
#define SDL_Log(msg, ...)
#define SDL_LogWarn(a, msg, ...)
#define SDL_LogError(a, msg, ...)
#endif

// simulation of Windows GetTickCount()
static unsigned long long
GetTickCountPortable()
{
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// Clock built upon Windows GetTickCount()
struct TickCountClock
{
  typedef unsigned long long                       rep;
  typedef std::milli                               period;
  typedef std::chrono::duration<rep, period>       duration;
  typedef std::chrono::time_point<TickCountClock>  time_point;
  static const bool is_steady = true;

  static time_point now() noexcept
  {
    return time_point(duration(GetTickCountPortable()));
  }
};

struct TCSDLTexture
{
  AVFrame *pTexture;
  uint32_t updateTime;
};

class TCVideoController
{
public:
  TCVideoController();
  virtual ~TCVideoController();
};

