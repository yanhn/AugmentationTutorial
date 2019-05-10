//
// Created by nvidia on 19-5-7.
//

#include "BasicAug.h"

namespace hawk{
    ImageData::ImageData(){
        img = cv::Mat();
        labelId = -1;
        boxInfos = {};
    }

    BasicAugmentation::BasicAugmentation(){
        _augmentationOpName = "BasicAug";
    }

    BasicAugmentation::~BasicAugmentation(){
    }

    bool BasicAugmentation::transform(ImageData& imgData){
        return false;
    }

    std::string BasicAugmentation::getTransformName(){
        return _augmentationOpName;
    }

    // StableResizeAugmentation
    StableResizeAugmentation::StableResizeAugmentation(int outWidth, int outHeight){
        _outWidth = outWidth;
        _outHeight = outHeight;
        _augmentationOpName = "StableResizeAug";
    }

    StableResizeAugmentation::~StableResizeAugmentation(){

    }

    bool StableResizeAugmentation::transform(ImageData& imgData){
        int inputWidth = imgData.img.cols;
        int inputHeight = imgData.img.rows;
        float ratioWidth = (float)inputWidth / _outWidth;
        float ratioHeight = (float)inputHeight / _outHeight;
        if (ratioWidth >= ratioHeight){
            int resizedHeight = int(inputHeight / ratioWidth);
            int heightPadding = (_outHeight - resizedHeight) / 2;
            cv::resize(imgData.img, imgData.img, cv::Size2d(_outWidth, resizedHeight));
            cv::copyMakeBorder(imgData.img, imgData.img, heightPadding, _outHeight - resizedHeight - heightPadding, 0, 0, cv::BORDER_CONSTANT);
            return true;
        } else{
            int resizedWidth = int(inputWidth / ratioHeight);
            int widthPadding = (_outWidth - resizedWidth) / 2;
            cv::resize(imgData.img, imgData.img, cv::Size2d(resizedWidth, _outHeight));
            cv::copyMakeBorder(imgData.img, imgData.img, 0, 0, widthPadding, _outWidth - resizedWidth - widthPadding, cv::BORDER_CONSTANT);
            return true;
        }
    }

    // HardResizeAugmentation
    HardResizeAugmentation::HardResizeAugmentation(int outWidth, int outHeight){
        _outWidth = outWidth;
        _outHeight = outHeight;
        _augmentationOpName = "HardResizeAug";
    }

    HardResizeAugmentation::~HardResizeAugmentation(){

    }

    bool HardResizeAugmentation::transform(ImageData& imgData){
        cv::resize(imgData.img, imgData.img, cv::Size2d(_outWidth, _outHeight));
        return true;
    }

    // SubMeanDivideVarAugmentation
    SubMeanDivideVarAugmentation::SubMeanDivideVarAugmentation(std::vector<float> means, std::vector<float> vars){
        assert(means.size() == vars.size());
        assert(means.size() >= 1);
        if (means.size() > 1){
            _isSingleChannel = false;
        } else {
            _isSingleChannel = true;
        }
        _means = means;
        _vars = vars;
        _augmentationOpName = "SubMeanDivideVarsAug";
    }

    SubMeanDivideVarAugmentation::~SubMeanDivideVarAugmentation(){

    }

    /*
     * input: imgData with img of type CV_8UC(n); after transform, change to CV_32FC(n)
     */
    bool SubMeanDivideVarAugmentation::transform(ImageData& imgData){
        if (_isSingleChannel){
            // Usually we want to do sub mean first then divide vars. But in this api, it do divide first. So we modify the mean by divide it.
            imgData.img.convertTo(imgData.img, CV_32FC1, 1/_vars[0], -_means[0] / _vars[0]);
            return true;
        }else{
            std::vector<cv::Mat> bgrPlanes;
            cv::split(imgData.img, bgrPlanes);
            for (int i = 0; i < bgrPlanes.size(); i ++ ){
                bgrPlanes[i].convertTo(bgrPlanes[i], CV_32FC1, 1/_vars[i], -_means[i]);
            }
            cv::merge(bgrPlanes, imgData.img);
            return true;
        }
    }
}