#include <iostream>
#include "AutoCalib.hpp"
#include "SurroundView.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>


bool SurroundView::init(const std::vector<cv::cuda::GpuMat>& imgs){
	
	if (isInit){
	    std::cerr << "SurroundView already initialize...\n";
	    return false;
	}
	
	imgs_num = imgs.size();	

	if (imgs_num <= 1){
	    std::cerr << "Not enough images in imgs vector, must be >= 2...\n";
	    return false;
	}
	
	cv::Size img_size = imgs[0].size();
	
	std::vector<cv::Mat> cpu_imgs(imgs_num);
	for (size_t i = 0; i < imgs_num; ++i){
	    imgs[i].download(cpu_imgs[i]);
	}


	AutoCalib autcalib(imgs_num);
	bool res = autcalib.init(cpu_imgs);
	if (!res){
	    std::cerr << "Error can't autocalibrate camera parameters...\n";
	    return false;
	}
	warped_image_scale = autcalib.get_warpImgScale();
	cameras = autcalib.getExtCameraParam();
	Ks_f = autcalib.getIntCameraParam();


	res = warpImage(cpu_imgs);
	if (!res){
	    std::cerr << "Error can't build warp images...\n";
	    return false;
	}
#ifdef CUT_OFF_FRAME
        res = prepareCutOffFrame(cpu_imgs);
        if (!res){
            std::cerr << "Error can't prepare blending ROI rect...\n";
            return false;
        }
#endif

	cuBlender = std::make_shared<CUDAFeatherBlender>(sharpness);
	cuBlender->prepare(corners, sizes, gpu_seam_masks);

	isInit = true;


	return isInit;
}



bool SurroundView::warpImage(const std::vector<cv::Mat>& imgs)
{
        gpu_seam_masks = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        corners = std::move(std::vector<cv::Point>(imgs_num));
        sizes = std::move(std::vector<cv::Size>(imgs_num));
        texXmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        texYmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

	/* warped images and masks */
	std::vector<cv::UMat> masks_warped_(imgs_num);
	std::vector<cv::UMat> imgs_warped(imgs_num);
	std::vector<cv::UMat> imgs_warped_f(imgs_num);
	std::vector<cv::Mat> masks(imgs_num);
	std::vector<cv::cuda::GpuMat> gpu_warpmasks(imgs_num);


	for (size_t i = 0; i < imgs_num; ++i){
	      masks[i].create(imgs[i].size(), CV_8U);
	      masks[i].setTo(cv::Scalar::all(255));
	}



        cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::PlaneWarper>();
        //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::SphericalWarper>();
        //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(2.f, 1.f);

        cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * work_scale));


        for(size_t i = 0; i < imgs_num; ++i){
              corners[i] = warper->warp(imgs[i], Ks_f[i], cameras[i].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, imgs_warped[i]);
              sizes[i] = imgs_warped[i].size();
              warper->warp(masks[i], Ks_f[i], cameras[i].R, cv::INTER_NEAREST, cv::BORDER_REFLECT, masks_warped_[i]);
              gpu_warpmasks[i].upload(masks_warped_[i]);
        }

	for(auto& msk : masks_warped_){
	      if (msk.cols > mask_maxnorm_size.width || msk.rows > mask_maxnorm_size.height ||
		  msk.cols < mask_minnorm_size.width || msk.rows < mask_minnorm_size.height) {
		      std::cerr << "Error: fail build masks for seam...\n";
		      return false;
	      }
	}


	compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
	compens->feed(corners, imgs_warped, masks_warped_);
	for (int i = 0; i < imgs_num; ++i){
	      compens->apply(i, corners[i], imgs_warped[i], masks_warped_[i]);
	      imgs_warped[i].convertTo(imgs_warped_f[i], CV_32F);
	}


	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::VORONOI_SEAM);


	seam_finder->find(imgs_warped_f, corners, masks_warped_);


	cv::Mat morphel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, masks_warped_[0].type(), morphel);

	cv::cuda::GpuMat tempmask, gpu_dilate_mask, gpu_seam_mask;
	cv::Mat xmap, ymap;
	for(size_t i = 0; i < imgs_num; ++i){
		tempmask.upload(masks_warped_[i]);
		dilateFilter->apply(tempmask, gpu_dilate_mask);
		cv::cuda::resize(gpu_dilate_mask, gpu_seam_mask, tempmask.size());
		cv::cuda::bitwise_and(gpu_seam_mask, gpu_warpmasks[i], gpu_seam_masks[i]);
		warper->buildMaps(imgs[i].size(), Ks_f[i], cameras[i].R, xmap, ymap);
		texXmap[i].upload(xmap);
		texYmap[i].upload(ymap);
	}


	if (!prepareGainMatrices(imgs_warped)){
	    std::cerr << "Error: fail build gain compensator matrices...\n";
	    return false;
	}

	return true;
}


#ifdef CUT_OFF_FRAME
bool SurroundView::prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs)
{
          cv::detail::MultiBandBlender blender(true, 5);

          blender.prepare(cv::detail::resultRoi(corners, sizes));

          cv::cuda::GpuMat warp_, warp_s, warp_img;
          for(size_t i = 0; i < imgs_num; ++i){
                  warp_.upload(cpu_imgs[i]);
                  cv::cuda::remap(warp_, warp_img, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);
                  //applyGpuCompensator(warp_img, gpu_gain_map[i]);
                  warp_img.convertTo(warp_s, CV_16S);
                  blender.feed(warp_s, gpu_seam_masks[i], corners[i]);
          }

          cv::Mat result, mask;
          blender.blend(result, mask);
          result.convertTo(result, CV_8U);

          cv::Mat thresh;
          cv::cvtColor(result, thresh, cv::COLOR_RGB2GRAY);
          cv::threshold(thresh, thresh, 32, 255, cv::THRESH_BINARY);
          cv::Canny(thresh, thresh, 1, 255);

          cv::morphologyEx(thresh, thresh, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
          cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

          std::vector<std::vector<cv::Point>> cnts;
          cv::findContours(thresh, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

          if (cnts.empty()){
                    std::cerr << "Error find contours for build rect panorama...\n:";
                    return false;
          }


          std::vector<cv::Point> _cnts;
          auto size_cnt = 1;
          auto total_idx = 0, pidx = 0;
          for(auto& pcnts : cnts){
              if (pcnts.size() > size_cnt){
                  size_cnt = pcnts.size();
                  pidx = total_idx;
              }
              total_idx += 1;
          }


          _cnts = cnts[pidx];
          if (size_cnt <= 1){
                    std::cerr << "Error find good contours...\n:";
                    return false;
          }


          std::sort(_cnts.begin(), _cnts.end(),
                    [](const cv::Point& l, const cv::Point& r){
                     if (l.x < r.x)
                         return true;
                     else if(l.x == r.x){
                         if (l.y < r.y)
                           return true;
                     }
                     return false;
          });

          cv::Point middle_bot, middle_top;
          cv::Mat res_col;
          auto x_ = 0;
          if (_cnts.size() % 2){
              auto x_ = _cnts[_cnts.size() / 2].x;
              res_col = thresh.col(x_);
              middle_bot.x = x_;
              middle_top.x = x_;
          }
          else{
              auto x_ = _cnts[_cnts.size() / 2 + 1].x;
              res_col = thresh.col(x_);
          }



          cv::drawContours(result, cnts, -1, cv::Scalar(0, 0, 255), 2);
          cv::circle(result, middle_bot, 10, cv::Scalar(255, 255, 255), -1);
          cv::circle(result, middle_top, 10, cv::Scalar(255, 255, 255), -1);
          cv::imshow("Camera - 2:", result);

          return false;


          //constexpr auto tb_remove = 0.01;
          cv::Rect boundRect = cv::Rect(0, 0, result.cols-1, result.rows-1);
          //auto rem_offset = boundRect.height * tb_remove;

          //blendingRect = cv::Rect(boundRect.x, boundRect.y + rem_offset, boundRect.width, boundRect.height - rem_offset);


          //result = result(cv::Range(tl.x, result.rows - 1), cv::Range(tl.y, result.rows - 1));


          return true;
}
#endif

bool SurroundView::prepareGainMatrices(const std::vector<cv::UMat>& warp_imgs)
{
	std::vector<cv::Mat> gain_map;

	compens->getMatGains(gain_map);

	if (gain_map.size() == 0){
		std::cerr << "Error: no gain matrices for exposure compensator...\n";
		return false;
	}
	if (gain_map.size() != imgs_num){
		std::cerr << "Error: wrong size gain matrices for exposure compensator...\n";
		return false;
	}

	gpu_gain_map = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

	for (size_t i = 0; i < imgs_num; ++i){
	      cv::resize(gain_map[i], gain_map[i], warp_imgs[i].size(), 0., 0., cv::INTER_LINEAR);

	      if (gain_map[i].channels() != 3){
		    std::vector<cv::Mat> gains_channels;
		    gains_channels.push_back(gain_map[i]);
		    gains_channels.push_back(gain_map[i]);
		    gains_channels.push_back(gain_map[i]);
		    cv::merge(gains_channels, gain_map[i]);
	      }
	      gpu_gain_map[i].upload(gain_map[i]);
	}


	gain_map.clear();

	return true;
}






bool SurroundView::stitch(const std::vector<cv::cuda::GpuMat*>& imgs, cv::cuda::GpuMat& blend_img)
{
    if (!isInit){
        std::cerr << "SurroundView was not initialized...\n";
        return false;
    }

    cv::cuda::GpuMat gpuimg_warped_s, gpuimg_warped;
    cv::cuda::GpuMat stitch, mask_;

    #pragma omp parallel for default(none) shared(imgs) private(gpuimg_warped, gpuimg_warped_s)
    for(size_t i = 0; i < imgs_num; ++i){

          cv::cuda::remap(*imgs[i], gpuimg_warped, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);

          gpuimg_warped.convertTo(gpuimg_warped_s, CV_16S, streamObj);

          cuBlender->feed(gpuimg_warped_s, gpu_seam_masks[i], corners[i], i, streamObj);

    }

    cuBlender->blend(stitch, mask_, streamObj);

#ifdef COLOR_CORRECTION

    stitch.convertTo(gpuimg_warped, CV_8U, streamObj);

    cv::cuda::cvtColor(gpuimg_warped, gpuimg_warped, cv::COLOR_RGB2YCrCb, 0, streamObj);

    cv::cuda::split(gpuimg_warped, inrgb, streamObj);

    cv::cuda::equalizeHist(inrgb[0], inrgb[0], streamObj);

    cv::cuda::merge(inrgb, gpuimg_warped, streamObj);

    cv::cuda::cvtColor(gpuimg_warped, blend_img, cv::COLOR_YCrCb2RGB, 0, streamObj);

#else
    //cv::cuda::GpuMat temp;
    stitch.convertTo(blend_img, CV_8U, streamObj);
   // blend_img = temp(blendingRect);
#endif
    return true;
}





void SurroundView::applyGpuCompensator(cv::cuda::GpuMat& _image, cv::cuda::GpuMat& gpu_gain_map)
{
	CV_Assert(_image.type() == CV_8UC3);

	cv::cuda::GpuMat temp;

	_image.convertTo(temp, CV_32F, streamObj);

	cv::cuda::multiply(temp, gpu_gain_map, temp, 1, CV_32F, streamObj);

	temp.convertTo(_image, CV_8U, streamObj);
}







