//
// Created by nvidia on 19-5-7.
//

#ifndef TRTINFER_BASICAUG_H
#define TRTINFER_BASICAUG_H

#include <vector>
#include <opencv.hpp>

namespace hawk{
    /*
     * Description: image data struct
     * labelId: used in image classification application.
     * boxInfos: used in detection application, 2d vector, inner vector in format of: [xmin, ymin, xmax, ymax, classId]
     * img: cv::Mat image
     */
    struct ImageData{
        int labelId;
        std::vector<std::vector<float>> boxInfos;
        cv::Mat img;
        ImageData();
    };

    /*
     * Description: data augmentation interface. It is a pure virtual class.
     * Function: transform is a pure virtual method, with 1 in-out parameter.
     */
    class BasicAugmentation{
    public:
        BasicAugmentation();
        virtual ~BasicAugmentation();
        virtual bool transform(ImageData& imgData);
        std::string getTransformName();
    protected:
        std::string _augmentationOpName;
    };

    /*
     * Description: Used to do image resize. Stable means keep w/h ratio unchanged, so no distortion introduced
     * Constructor:
     */
    class StableResizeAugmentation:BasicAugmentation{
    public:
        StableResizeAugmentation(int outWidth, int outHeight);
        ~StableResizeAugmentation();
        bool transform(ImageData& imgData);
    private:
        int _outWidth;
        int _outHeight;
    };

    /*
  * Description: Used to do image resize. hard resize with w/h ratio changed, so distortion introduced
  * Constructor:
  */
    class HardResizeAugmentation:BasicAugmentation{
    public:
        HardResizeAugmentation(int outWidth, int outHeight);
        ~HardResizeAugmentation();
        bool transform(ImageData& imgData);
    private:
        int _outWidth;
        int _outHeight;
    };

    /*
     * Description: Used to do sub mean and divide vars oprations.
     * Convert input rtype from
     */
    class SubMeanDivideVarAugmentation:BasicAugmentation{
    public:
        SubMeanDivideVarAugmentation(std::vector<float> means, std::vector<float> vars);
        ~SubMeanDivideVarAugmentation();
        bool transform(ImageData& imgData);
    private:
        std::vector<float> _means;
        std::vector<float> _vars;
        bool _isSingleChannel;
    };
}

#endif //TRTINFER_BASICAUG_H