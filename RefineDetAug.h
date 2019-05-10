//
// Created by nvidia on 19-5-7.
//

#ifndef TRTINFER_REFINEDETAUG_H
#define TRTINFER_REFINEDETAUG_H

#include<dataAugmentation/BasicAug.h>

namespace hawk{
    class RefineDetAug{
    public:
        RefineDetAug();
        ~RefineDetAug();
        bool tranform(ImageData& imageData);

    private:
        std::vector<BasicAugmentation*> _augs;
    };

    void testRefineDetAug();
}

#endif //TRTINFER_REFINEDETAUG_H