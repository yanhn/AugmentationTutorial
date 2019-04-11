'''
@author: double4tar
@contact: double4tar@gmail.com
@file: augmentation.py
@time: 19-04-10
@desc: modify & copy crom https://github.com/amdegroot/ssd.pytorch
'''

import cv2
import numpy as np
from numpy import random
# import torch
# from torchvision import transforms

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class RandomHorizontalMirror(object):
    """
    do horizontal mirror, flip the img, nothing to do with label,
    newXmin = width - oriXmax
    newXmax = width - oriXmin
    """
    def __call__(self, image, boxes, label):
        width = image.shape[1]
        if random.randint(2):
            image = cv2.flip(image, 1)
            if boxes is not None:
                boxesTmp = boxes.copy()
                boxes[:, 0::2] = width - boxesTmp[:, 2::-2]
        return image, boxes, label

class RandomVerticalMirror(object):
    """
    do vertical mirror, flip the img, nothing to do with label,
    newYmin = height - oriYmax
    newYmax = height - oriYmin
    """
    def __call__(self, image, boxes, label):
        height = image.shape[0]
        if random.randint(2):
            image = cv2.flip(image, 0)
            if boxes is not None:
                boxesTmp = boxes.copy()
                boxes[:, 1::2] = height - boxesTmp[:, 3::-2]
        return image, boxes, label

class Rotate90(object):
    """
    do random roate 90 degree, nothing to do with label, but not very useful in classification
    clockWise->boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = height - oriBox[:, 3], oriBox[:, 0], height - oriBox[:, 1], oriBox[:, 2]
    counter clockWise->boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = oriBox[:, 1], width - oriBox[:, 2], oriBox[:, 3], width - oriBox[:, 0]
    """
    def __call__(self, image, boxes, label):
        height, width = image.shape[:2]
        oriBox = boxes.copy()
        if random.randint(2):
            if random.randint(2):
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if boxes is not None:
                    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = height - oriBox[:, 3], oriBox[:, 0], height - oriBox[:, 1], oriBox[:, 2]
            else:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if boxes is not None:
                    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = oriBox[:, 1], width - oriBox[:, 2], oriBox[:, 3], width - oriBox[:, 0]
        return image, boxes, label

class ClipBox(object):
    """
    restrict bboxes in [0, width] and [0, height]
    """
    def __call__(self, image, boxes, label):
        if boxes is not None:
            height, width = image.shape[:2]
            boxes[:, 0] = np.minimum(np.maximum(0, boxes[:, 0]), width)
            boxes[:, 1] = np.minimum(np.maximum(0, boxes[:, 1]), height)
            boxes[:, 2] = np.minimum(np.maximum(0, boxes[:, 2]), width)
            boxes[:, 3] = np.minimum(np.maximum(0, boxes[:, 3]), height)
        return image, boxes, label

# TODO
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, leastBorderGap = 0):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self._leastBorderGap = leastBorderGap

    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        insterestedCanvas = np.array([int(width * self._leastBorderGap), int(height * self._leastBorderGap),
                                      width - int(width * self._leastBorderGap), height - int(height * self._leastBorderGap)])
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # not used for now
                # focusedOverlap = interOverRectA(boxes, insterestedCanvas)
                # if(focusedOverlap.min() < 0.6):
                #     continue

                # cut the crop from the image
                try:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                except:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # use full box to filter instead of center
                m1HasPartsImgButNoBox = []
                for i in range(centers.shape[0]):
                    if (m1[i]):
                        m1HasPartsImgButNoBox.append(False)
                    else:
                        if(rect[0] < boxes[i, 2] and rect[1] < boxes[i, 3]):
                            m1HasPartsImgButNoBox.append(True)
                        else:
                            m1HasPartsImgButNoBox.append(False)
                m2HasPartsImgButNoBox = []
                for i in range(centers.shape[0]):
                    if (m2[i]):
                        m2HasPartsImgButNoBox.append(False)
                    else:
                        if(rect[0] > boxes[i, 0] and rect[1] > boxes[i, 1]):
                            m2HasPartsImgButNoBox.append(True)
                        else:
                            m2HasPartsImgButNoBox.append(False)
                if(np.array(m1HasPartsImgButNoBox).any() or np.array(m2HasPartsImgButNoBox).any()):
                    continue

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # special case: filter all parts cls (idx = 2 for pedestrian_side)
                idxInUse = labels[mask]
                cmpResult = (idxInUse == 2)
                if (cmpResult.all()):
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

            # logging.info("Run out of 50 tries during RandomSampleCrop.")
            return image, boxes, labels

# TODO
class Expand(object):
    #make border
    def __init__(self, mean, maxExpandRatio = 2):
        self.mean = mean
        self._maxExpandRatio = maxExpandRatio

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        try:
            height, width, depth = image.shape
            ratio = random.uniform(1, self._maxExpandRatio)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = image
            image = expand_image
        except:
            height, width = image.shape
            ratio = random.uniform(1, self._maxExpandRatio)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio)),
                dtype=image.dtype)
            expand_image[:, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = image
            image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

# TODO comment
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class StableResize(object):
    """
    Stable resize keep width / height ratio unchanged.
    """
    def __init__(self, size=300, mean=(0,0,0)):
        self.size = size
        self.mean = mean

    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        maxSide = np.maximum(height, width)
        ratio = self.size / float(maxSide)
        newShape = (int(width * ratio), int(height * ratio))        # (w, h)
        image = cv2.resize(image, newShape)

        diffX = self.size - image.shape[1]
        diffY = self.size - image.shape[0]

        padX = np.random.randint(0, diffX) if diffX > 0 else 0
        padY = np.random.randint(0, diffY) if diffY > 0 else 0
        image = cv2.copyMakeBorder(image, padY, diffY - padY, padX, diffX - padX, cv2.BORDER_CONSTANT, self.mean)
        if boxes is not None:
            boxes[:, 0] = boxes[:, 0] * ratio + padX
            boxes[:, 1] = boxes[:, 1] * ratio + padY
            boxes[:, 2] = boxes[:, 2] * ratio + padX
            boxes[:, 3] = boxes[:, 3] * ratio + padY

        return image, boxes, labels

# TODO
# class FFTTrans(object):
#     # FFT must do after resize and random crop
#     def __call__(self, image, boxes=None, labels=None):
#         imgImgPart = np.zeros(image.shape)
#         imgComplex = np.zeros((image.shape[0], image.shape[1], 2))
#         imgComplex[:, :, 0] = image
#         imgComplex[:, :, 1] = imgImgPart
#
#
#         x = torch.Tensor(imgComplex)
#         y = torch.fft(x, 2, True)
#
#         normY = y.cpu().numpy()
#         normY = (normY - normY.mean()) / normY.var()
#         normY = np.transpose(normY, (2, 0, 1))
#         ret = np.concatenate((image.reshape(1, image.shape[0], image.shape[1]), normY))
#         # shape 300 x 300 x 3
#         return ret, boxes, labels

class ConverBGRToGray(object):
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels

class Transpose201(object):
    def __call__(self, image, boxes=None, labels=None):
        if len(image.shape) == 2:
            # special case for grayscale image
            image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1))
        return image, boxes, labels

def displayData(image, boxes, label):
    imageTmp = image.copy()
    for id in range(boxes.shape[0]):
        cv2.rectangle(imageTmp, (int(boxes[id][0]), int(boxes[id][1])), (int(boxes[id][2]), int(boxes[id][3])), (0, 255, 255), 2)
        cv2.putText(imageTmp, "{}".format(label[id]), (int(boxes[id][0]), int(boxes[id][1]) - 1), 1, 1, (0, 0, 255))
    cv2.imshow("data", imageTmp)
    cv2.waitKey()

class ConvertFromIntToFloat(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertFromFloatToInt(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.uint8), boxes, labels

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123), std=(1,1,1)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            # random contrast, hue, saturation
            RandomSampleCrop(),
            Expand(self.mean),
            RandomHorizontalMirror(),
            RandomVerticalMirror(),
            Rotate90(),
            StableResize(self.size, self.mean),
            SubtractMeans(self.mean), # grayscale rgb mean wrong
            ClipBox(),
            ToPercentCoords(),        #
            Transpose201()
        ])

    def __call__(self, img, boxes, labels = None):
        img, boxes, labels = self.augment(img, boxes, labels)
        # displayData(img, boxes, labels)
        return img, boxes, labels

class SSDAugmentationGray(object):
    def __init__(self, size=300, mean=(120), std=(1)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConverBGRToGray(),
            # random contrast, hue, saturation
            RandomSampleCrop(),
            Expand(self.mean),
            RandomHorizontalMirror(),
            RandomVerticalMirror(),
            Rotate90(),
            StableResize(self.size, self.mean),
            SubtractMeans(self.mean), # grayscale rgb mean wrong
            ClipBox(),
            ToPercentCoords(),        #
            Transpose201()
        ])

    def __call__(self, img, boxes, labels = None):
        img, boxes, labels = self.augment(img, boxes, labels)
        # displayData(img, boxes, labels)
        return img, boxes, labels

class SSDAugmentationTest(object):
    def __init__(self, size=300, mean=(120), std=(1)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            ConverBGRToGray(),
            # random contrast, hue, saturation
            StableResize(self.size, self.mean),
            SubtractMeans(self.mean), # grayscale rgb mean wrong
            ToPercentCoords(),        #
            Transpose201()
        ])

    def __call__(self, img, boxes, labels = None):
        img, boxes, labels = self.augment(img, boxes, labels)
        # displayData(img, boxes, labels)
        return img, boxes, labels