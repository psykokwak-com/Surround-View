#include "TCVideoClient.h"

//#define WITHAUDIO

TCVideoClientTimeoutHandler::TCVideoClientTimeoutHandler()
{
  _timeout = 5000; // in ms
  if (_timeout < 0) _timeout = 0;

  reset();
}

void TCVideoClientTimeoutHandler::reset()
{
  _last = GetTickCountPortable();
}

bool TCVideoClientTimeoutHandler::isTimeout()
{
  uint32_t actualDelay = GetTickCountPortable() - _last;
  return actualDelay > _timeout;
}

int TCVideoClientTimeoutHandler::getTimeout()
{
  return _timeout;
}

int TCVideoClientTimeoutHandler::checkInterrupt(void *ctx)
{
  TCVideoClient *vc = static_cast<TCVideoClient *>(ctx);
  return !vc || !vc->running() || vc->getTimeoutHandler()->isTimeout();
}


static enum AVPixelFormat _getHwFmt(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
  const enum AVPixelFormat *p;

  TCVideoClient *o = static_cast<TCVideoClient*>(ctx->opaque);

  for (p = pix_fmts; *p != -1; p++) {
    if (*p == o->_hwPixFmt)
      return *p;
  }

  SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "TCVideoClient : Failed to get HW surface format.");

  return AV_PIX_FMT_NONE;
}



static int _threadfn(void *ctx)
{
  TCVideoClient *o = static_cast<TCVideoClient*>(ctx);
  return o->worker();
}


TCVideoClient::TCVideoClient(const std::string &name)
  : _pVideoThread(NULL)
  , _videoThreadRunning(false)
  , _url("")
  , _name(name)
  , _rtspTimeout(-1)
  , _transport("")
{
  _texture.pTexture = NULL;
  _texture.updateTime = 0;
  SDL_Log("TCVideoClient::TCVideoClient()");
}


TCVideoClient::~TCVideoClient()
{
  SDL_Log("TCVideoClient::~TCVideoClient() :: begin");
  stop();
  SDL_Log("TCVideoClient::~TCVideoClient() :: end");
}

void TCVideoClient::setUrl(const std::string &url)
{
  _url = url;
}

// https://github.com/rambodrahmani/ffmpeg-video-player/blob/master/tutorial02/tutorial02.c
// https://gist.github.com/MarcoQin/d7d935e87a44410966ed8cad066953bc <= With audio
TCVideoClientTimeoutHandler *TCVideoClient::getTimeoutHandler()
{
  return &_videoTimeoutHandler;
}

int TCVideoClient::worker()
{
  SDL_Log("TCVideoClient::worker() :: begin");

  _videoThreadRunning = true;

  while (42) {
    if (workerStream() == 0)
      break;

    if (_videoThreadRunning)
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  SDL_Log("TCVideoClient::worker() :: end");

  return 0;
}

int TCVideoClient::workerStream()
{
  AVFormatContext *pFormatCtx = NULL;
  AVDictionary *pOpts = NULL;
  int videoStream = -1;
  int audioStream = -1;
  const AVCodec *pVCodec = NULL;
  AVCodecContext *pVCodecCtx = NULL;
  const AVCodec *pACodec = NULL;
  AVCodecContext *pACodecCtx = NULL;
  AVFrame *pFrame = NULL;
  AVFrame *pFrameReceived = NULL;
  AVPacket *pPacket = NULL;
  int numBytes = 0;
  struct SwsContext *pSwsCtx = NULL;
  struct SwrContext *pSwrCtx = NULL;
  uint8_t *pBuffer = NULL;
  AVFrame *pFrameSws = NULL;
  std::string timeout;
  std::string buffersize;
  int *pRTSPStreamTimeout_HACK = NULL;

  std::string hwTypeName;
  AVHWDeviceType hwType = AV_HWDEVICE_TYPE_NONE;
  AVBufferRef *pHwDeviceCtx = NULL;
  AVFrame *pFrameDecoded = NULL;

  bool useSws = false;

  int ret = -1;


  SDL_Log("TCVideoClient::workerLoop() :: begin");

  pFormatCtx = avformat_alloc_context();
  if (pFormatCtx == NULL) {
    goto fail;
  }

  pFormatCtx->flags |= AVFMT_FLAG_NONBLOCK | AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS | AVFMT_FLAG_DISCARD_CORRUPT;
  pFormatCtx->interrupt_callback.callback = &TCVideoClientTimeoutHandler::checkInterrupt;
  pFormatCtx->interrupt_callback.opaque = this;

  timeout = std::to_string(_videoTimeoutHandler.getTimeout() * 1000);
  buffersize = std::to_string(1 * 1024 * 1024); // 1MB

  av_dict_set(&pOpts, "stimeout", timeout.c_str(), 0); // in us
  av_dict_set(&pOpts, "buffer_size", buffersize.c_str(), 0); // in bytes (udp)
  av_dict_set(&pOpts, "recv_buffer_size", buffersize.c_str(), 0); // in bypte (tcp)
  av_dict_set(&pOpts, "tcp_nodelay", "1", 0);

  if (_transport.size() > 0)
    av_dict_set(&pOpts, "rtsp_transport", _transport.c_str(), 0);

  _videoTimeoutHandler.reset();
  ret = avformat_open_input(&pFormatCtx, _url.c_str(), NULL, &pOpts);
  if (ret < 0) {
    goto fail;
  }

  ret = avformat_find_stream_info(pFormatCtx, NULL);
  if (ret < 0) {
    goto fail;
  }

  av_dump_format(pFormatCtx, 0, _url.c_str(), 0);

  // FUCKING HUGLY HACK to override the timeout delay from ffmpeg library
  // Force timeout at 10 seconds to send SDP keep-alive accordingly to the timeout parameters in url.
  // This hack can no longer work on ffmpeg update or on other architecture
  pRTSPStreamTimeout_HACK = (int*)((char*)(pFormatCtx->priv_data) + 0x234);
  SDL_Log("TCVideoClient : RTSP default timeout=%d", *pRTSPStreamTimeout_HACK);
  if (_rtspTimeout != -1)
  {
    *pRTSPStreamTimeout_HACK = _rtspTimeout;
    SDL_Log("TCVideoClient : RTSP forced timeout=%d", *pRTSPStreamTimeout_HACK);

  }

  for (unsigned int i = 0; i < pFormatCtx->nb_streams; i++)
  {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && videoStream < 0)
      videoStream = i;

    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audioStream < 0)
      audioStream = i;
  }

  if (videoStream == -1) {
    goto fail;
  }

  pVCodec = avcodec_find_decoder(pFormatCtx->streams[videoStream]->codecpar->codec_id);
  if (pVCodec == NULL) {
    goto fail;
  }

  SDL_Log("TCVideoClient : Camera %s, codec : %s : %s", _name.c_str(), pVCodec->name, pVCodec->long_name);

  // Hardware decoding
  // http://transit.iut2.univ-grenoble-alpes.fr/cgi-bin/dwww/usr/share/doc/ffmpeg/api/hw__decode_8c_source.html
  hwTypeName = ""; // auto TCConfigurator.application["camera"]["hwAccel"].get<std::string>();
  _hwPixFmt = AV_PIX_FMT_NONE;
  if (hwTypeName.size() > 0)
  {
    hwType = av_hwdevice_find_type_by_name(hwTypeName.c_str());
    if (hwType == AV_HWDEVICE_TYPE_NONE) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "TCVideoClient : Device type %s is not supported.", hwTypeName.c_str());
      SDL_Log("TCVideoClient : Available device types:");
      while ((hwType = av_hwdevice_iterate_types(hwType)) != AV_HWDEVICE_TYPE_NONE)
        SDL_Log("TCVideoClient : - %s", av_hwdevice_get_type_name(hwType));
      goto fail;
    }

    for (int i = 0;; i++)
    {
      const AVCodecHWConfig *config = avcodec_get_hw_config(pVCodec, i);
      if (!config) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "TCVideoClient : Decoder %s does not support device type %s.", pVCodec->name, av_hwdevice_get_type_name(hwType));
        goto fail;
      }
      if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == hwType) {
        _hwPixFmt = config->pix_fmt;
        break;
      }
    }
  }

  pVCodecCtx = avcodec_alloc_context3(pVCodec);
  if (pVCodecCtx == NULL) {
    goto fail;
  }

  pVCodecCtx->opaque = this;

  ret = avcodec_parameters_to_context(pVCodecCtx, pFormatCtx->streams[videoStream]->codecpar);
  if (ret != 0) {
    goto fail;
  }

  if (hwTypeName.size() > 0)
  {
    pVCodecCtx->get_format = _getHwFmt;

    ret = av_hwdevice_ctx_create(&pHwDeviceCtx, hwType, NULL, NULL, 0);
    if (ret < 0) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "TCVideoClient : Failed to create specified HW device.");
      goto fail;
    }

    pVCodecCtx->hw_device_ctx = av_buffer_ref(pHwDeviceCtx);
    if (!pVCodecCtx->hw_device_ctx) {
      goto fail;
    }
  }


  ret = avcodec_open2(pVCodecCtx, pVCodec, NULL);
  if (ret < 0) {
    goto fail;
  }

#ifdef WITHAUDIO
  // if stream has audio
  if (audioStream != -1)
  {
    pACodec = avcodec_find_decoder(pFormatCtx->streams[audioStream]->codecpar->codec_id);
    if (pACodec == NULL) {
      goto fail;
    }

    pACodecCtx = avcodec_alloc_context3(pACodec);
    if (pACodecCtx == NULL) {
      goto fail;
    }

    ret = avcodec_parameters_to_context(pACodecCtx, pFormatCtx->streams[audioStream]->codecpar);
    if (ret != 0) {
      goto fail;
    }

    ret = avcodec_open2(pACodecCtx, pACodec, NULL);
    if (ret < 0) {
      goto fail;
    }

    if (pACodecCtx->channels > 0 && pACodecCtx->channel_layout == 0)
      pACodecCtx->channel_layout = av_get_default_channel_layout(pACodecCtx->channels);
    if (pACodecCtx->channels == 0 && pACodecCtx->channel_layout > 0)
      pACodecCtx->channels = av_get_channel_layout_nb_channels(pACodecCtx->channel_layout);

    pSwrCtx = swr_alloc_set_opts(NULL,  // we're allocating a new context
      AV_CH_LAYOUT_STEREO,              // out_ch_layout
      AV_SAMPLE_FMT_S16,                // out_sample_fmt
      44100,                            // out_sample_rate
      pACodecCtx->channel_layout,       // in_ch_layout
      pACodecCtx->sample_fmt,           // in_sample_fmt
      pACodecCtx->sample_rate,          // in_sample_rate
      0,                                // log_offset
      NULL);                            // log_ctx

    if (!pSwrCtx || swr_init(pSwrCtx) < 0) {
      goto fail;
    }
  }
#endif

  pFrameReceived = av_frame_alloc();
  if (pFrameReceived == NULL) {
    goto fail;
  }

  pFrameDecoded = av_frame_alloc();
  if (pFrameDecoded == NULL) {
    goto fail;
  }

  pPacket = av_packet_alloc();
  if (pPacket == NULL) {
    goto fail;
  }

  while (_videoThreadRunning && av_read_frame(pFormatCtx, pPacket) == 0)
  {
    _videoTimeoutHandler.reset();

#ifdef WITHAUDIO
    if (pPacket->stream_index == audioStream)
    { // Process audio frame

      ret = avcodec_send_packet(pACodecCtx, pPacket);
      if (ret == AVERROR(EAGAIN)) {
        continue;
      }
      if (ret == AVERROR_EOF) {
        break;
      }
      if (ret == AVERROR(EINVAL)) {
        break;
      }
      if (ret == AVERROR(ENOMEM)) {
        break;
      }

      while (ret >= 0)
      {
        ret = avcodec_receive_frame(pACodecCtx, pFrameReceived);
        if (ret == AVERROR(EAGAIN)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          break;
        }
        if (ret == AVERROR_EOF) {
          break;
        }

        if (ret < 0) {
          goto fail;
        }

        int sampleSize = av_get_bytes_per_sample(pACodecCtx->sample_fmt);
        if (sampleSize < 0) {
          goto fail;
        }

        uint8_t *output = NULL;
        int dst_nb_samples = av_rescale_rnd(swr_get_delay(pSwrCtx, pFrameReceived->sample_rate) + pFrameReceived->nb_samples, 44100, pFrameReceived->sample_rate, AV_ROUND_UP);
        av_samples_alloc(&output, NULL, 2, dst_nb_samples, AV_SAMPLE_FMT_S16, 0);
        int nb = swr_convert(pSwrCtx, &output, dst_nb_samples, (const uint8_t**)pFrameReceived->data, pFrameReceived->nb_samples);
        int data_size = pFrameReceived->channels * nb * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
        _pVideoContext->getSDLContext()->getAudio()->addAudioSamplesToQueue(output, data_size);
        av_freep(&output);
      }
    }
#endif

    if (pPacket->stream_index == videoStream)
    { // Process video frame

      ret = avcodec_send_packet(pVCodecCtx, pPacket);
      if (ret == AVERROR(EAGAIN)) {
        continue;
      }
      if (ret == AVERROR_EOF) {
        break;
      }
      if (ret == AVERROR(EINVAL)) {
        break;
      }
      if (ret == AVERROR(ENOMEM)) {
        break;
      }


      while (ret >= 0)
      {
        ret = avcodec_receive_frame(pVCodecCtx, pFrameReceived);
        if (ret == AVERROR(EAGAIN)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          break;
        }
        if (ret == AVERROR_EOF) {
          break;
        }

        if (ret < 0) {
          goto fail;
        }

        pFrame = pFrameReceived;

        if (hwTypeName.size() > 0)
        {
          if (pFrameReceived->format == _hwPixFmt) {
            /* retrieve data from GPU to CPU */
            if ((ret = av_hwframe_transfer_data(pFrameDecoded, pFrameReceived, 0)) < 0) {
              SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "TCVideoClient : Error transferring the data to system memory");
              goto fail;
            }
            pFrame = pFrameDecoded;
          }
        }

        //useSws = _AVtoSDLPixFmt((AVPixelFormat)pFrame->format) == SDL_PIXELFORMAT_UNKNOWN ? true : false;
        useSws = true;

        if (useSws)
        {
          AVPixelFormat swsTypeOut = AV_PIX_FMT_BGR24;

          if (!pSwsCtx)
          {
            // set up our SWSContext to convert the image data
            pSwsCtx = sws_getContext(
              pVCodecCtx->width,
              pVCodecCtx->height,
              (AVPixelFormat)pFrame->format,
              pVCodecCtx->width,
              pVCodecCtx->height,
              swsTypeOut,
              SWS_FAST_BILINEAR,
              NULL,
              NULL,
              NULL
            );

            if (pSwsCtx == NULL) {
              goto fail;
            }

            numBytes = av_image_get_buffer_size(
              swsTypeOut,
              pVCodecCtx->width,
              pVCodecCtx->height,
              32
            );

            pBuffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
            if (pBuffer == NULL) {
              goto fail;
            }

            pFrameSws = av_frame_alloc();
            if (pFrameSws == NULL) {
              goto fail;
            }

            av_image_fill_arrays(
              pFrameSws->data,
              pFrameSws->linesize,
              pBuffer,
              swsTypeOut,
              pVCodecCtx->width,
              pVCodecCtx->height,
              32
            );

            SDL_Log("TCVideoClient : Camera %s : enable SWS to convert image from %d to %d", _name.c_str(), pFrame->format, swsTypeOut);
          }

          sws_scale(
            pSwsCtx,
            (uint8_t const * const *)pFrame->data,
            pFrame->linesize,
            0,
            pVCodecCtx->height,
            pFrameSws->data,
            pFrameSws->linesize
          );

          pFrameSws->width = pFrame->width;
          pFrameSws->height = pFrame->height;
          pFrameSws->format = swsTypeOut;
        }

        /*
        printf(
          "Frame %c (%d) pts %d dts %d key_frame %d [coded_picture_number %d, display_picture_number %d, %dx%d]\n",
          av_get_picture_type_char(pFrame->pict_type),
          pVCodecCtx->frame_number,
          pFrame->pts,
          pFrame->pkt_dts,
          pFrame->key_frame,
          pFrame->coded_picture_number,
          pFrame->display_picture_number,
          pVCodecCtx->width,
          pVCodecCtx->height
        );
        */

        {
          //TCCriticalSection cs(_pVideoContext->getSDLContext()->getWindow()->getMutexDisplay());

          //if (!_texture.pTexture)
          {
            if (useSws)
              ;//_texture.pTexture = SDL_CreateTexture(_pVideoContext->getSDLContext()->getWindow()->getRenderer(), SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, pVCodecCtx->width, pVCodecCtx->height);
            else
              ;//_texture.pTexture = SDL_CreateTexture(_pVideoContext->getSDLContext()->getWindow()->getRenderer(), _AVtoSDLPixFmt((AVPixelFormat)pFrame->format), SDL_TEXTUREACCESS_STREAMING, pVCodecCtx->width, pVCodecCtx->height);

            //_texture.updateTime = 0;

            //if (_texture.pTexture == NULL) {
            //  goto fail;
            //}
          }

          //if (_texture.pTexture)
          {
            _mutexVideoThread.lock();
            if (useSws)
              _texture.pTexture = pFrameSws;// SDL_UpdateTexture(_texture.pTexture, NULL, pFrameSws->data[0], pFrameSws->linesize[0]);
            else
              _texture.pTexture = pFrame;
            _mutexVideoThread.unlock();
            /*
              switch (pFrame->format)
              {
              case AV_PIX_FMT_NV12:
              case AV_PIX_FMT_NV24:
                SDL_UpdateNVTexture(_texture.pTexture,
                  NULL,
                  pFrame->data[0],
                  pFrame->linesize[0],
                  pFrame->data[1],
                  pFrame->linesize[1]
                );
                break;
              case AV_PIX_FMT_YUVJ420P:
              case AV_PIX_FMT_YUV420P:
              case AV_PIX_FMT_YUVJ422P:
              case AV_PIX_FMT_YUV422P:
              case AV_PIX_FMT_YUVJ444P:
              case AV_PIX_FMT_YUV444P:
                SDL_UpdateYUVTexture(
                  _texture.pTexture,
                  NULL,
                  pFrame->data[0],
                  pFrame->linesize[0],
                  pFrame->data[1],
                  pFrame->linesize[1],
                  pFrame->data[2],
                  pFrame->linesize[2]
                );
                break;
              default:
                SDL_UpdateTexture(_texture.pTexture, NULL, pFrame->data[0], pFrame->linesize[0]);
              }
              */
            _texture.updateTime = GetTickCountPortable();
          }
        }
        //_pVideoContext->getSDLContext()->getWindow()->requestUpdate();
      }
    }

    av_packet_unref(pPacket);
  }

fail:

  _texture.pTexture = NULL;
  _texture.updateTime = 0;
 
  av_frame_free(&pFrameSws);
  av_free(pBuffer);
  sws_freeContext(pSwsCtx);
  swr_free(&pSwrCtx);
  av_packet_free(&pPacket);
  av_frame_free(&pFrameReceived);
  av_frame_free(&pFrameDecoded);
  avcodec_free_context(&pVCodecCtx);
  avcodec_free_context(&pACodecCtx);
  avformat_close_input(&pFormatCtx);
  av_dict_free(&pOpts);
  avformat_free_context(pFormatCtx);
  av_buffer_unref(&pHwDeviceCtx);

  SDL_Log("TCVideoClient::workerLoop() :: end");

  return _videoThreadRunning ? 1 : 0;
}

bool TCVideoClient::readLastFrame(cv::cuda::GpuMat &gpuFrame)
{
  _mutexVideoThread.lock();

  cv::Mat cpuMat(gpuFrame.size(), gpuFrame.type(), _texture.pTexture->data[0]);
  gpuFrame.upload(cpuMat);

  _mutexVideoThread.unlock();

  return true;
}

void TCVideoClient::setOverrideRTSPTimeout(int t)
{
  _rtspTimeout = t;
}

void TCVideoClient::setTransport(const std::string &transport)
{
  _transport = transport;
}


int TCVideoClient::play()
{
  //stop();

  SDL_Log("TCVideoClient::play() :: begin");

  if (!_pVideoThread)
    _pVideoThread = new std::thread(_threadfn, this);

  if (!_pVideoThread)
    return -1;

  SDL_Log("TCVideoClient::play() :: end");

  return 0;
}

int TCVideoClient::stop()
{
  if (!_pVideoThread)
    return -1;

  SDL_Log("TCVideoClient::stop() :: begin");

  _videoThreadRunning = false;
  if (_pVideoThread)
    _pVideoThread->join();
  _pVideoThread = NULL;

  SDL_Log("TCVideoClient::stop() :: end");

  return 0;
}

bool TCVideoClient::running()
{
  return _videoThreadRunning;
}

bool TCVideoClient::playing()
{
  return _videoThreadRunning && _texture.pTexture;
}


int TCVideoClient::getInfoResolutionX()
{
  int x = -1;

  //if (_texture.pTexture)
  //  SDL_QueryTexture(_texture.pTexture, NULL, NULL, &x, NULL);

  return x;
}

int TCVideoClient::getInfoResolutionY()
{
  int y = -1;

  //if (_texture.pTexture)
  //  SDL_QueryTexture(_texture.pTexture, NULL, NULL, NULL, &y);

  return y;
}

std::string TCVideoClient::getInfoName()
{
  return _name;
}
