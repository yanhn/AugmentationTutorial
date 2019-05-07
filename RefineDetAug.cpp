//
// Created by nvidia on 19-5-7.
//

#include "RefineDetAug.h"

namespace hawk{
    RefineDetAug::RefineDetAug(){
        _augs.push_back((BasicAugmentation*) new StableResizeAugmentation(320, 320));
        std::vector<float> means = {120.};
        std::vector<float> vars = {1.};
        _augs.push_back((BasicAugmentation*) new SubMeanDivideVarAugmentation(means, vars));
    }

    RefineDetAug::~RefineDetAug(){
        for(auto aug: _augs){
            delete(aug);
        }
    }

    bool RefineDetAug::tranform(ImageData& imageData){
        for (auto aug : _augs){
            if(!aug->transform(imageData)){
                std::cout << "Some aug wrong. " << aug->getTransformName() << std::endl;
                return false;
            }
        }
        return true;
    }

    void testRefineDetAug(){
        RefineDetAug refineAug;
        ImageData imgData;
        imgData.img = cv::imread("/home/nvidia/Pictures/2018090_21180.jpg", cv::IMREAD_GRAYSCALE);
        cv::imshow("before", imgData.img);
        bool ret = refineAug.tranform(imgData);
        imgData.img.convertTo(imgData.img, CV_8UC1);
        cv::imshow("after", imgData.img);
        cv::waitKey();
    }
}