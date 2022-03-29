# from torchvision.transforms import autoaugment, transforms
from torchvision.transforms import transforms
import numpy as np
import torch
import imgaug.augmenters as iaa
from torchvision.transforms import functional as F
import torchvision.transforms as T
from randaugment import CIFAR10Policy

class ClassificationPresetTrain:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hflip_prob=0.5,
                 auto_augment_policy=None, random_erase_prob=0.0):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
#         if auto_augment_policy is not None:
#             aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
#             trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)
    

class Augment(object):

    def __init__(self,num_of_augms=1):
        self.num_of_augms=num_of_augms
        self.seq =iaa.OneOf([
            iaa.Sequential([
                iaa.LinearContrast(alpha=(0.75, 1.25)),
                iaa.Fliplr(0.5),
                iaa.WithHueAndSaturation(
                iaa.WithChannels(0, iaa.Add((0, 50)))
            )
            ]),
            iaa.Sequential([
                iaa.Grayscale(alpha=(0.1, 0.3)),
                iaa.Fliplr(0.5),
                iaa.Affine(
                    translate_percent={"y": (-0.15, 0.15)}
                )
            ]),
            iaa.Sequential([
                iaa.imgcorruptlike.MotionBlur(severity=1),
                iaa.LinearContrast((0.6, 1.4)),
                iaa.ShearX((-10, 10))
            ]),
            iaa.Sequential([
                iaa.imgcorruptlike.GaussianNoise(severity=1),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                iaa.ShearY((-10, 10))
            ]),
            iaa.Sequential([
                iaa.Cutout(nb_iterations=(1, 2), size=0.1, squared=False),
                iaa.Multiply((0.8, 1.2), per_channel=0.25),
                iaa.Fliplr(0.5),
            ]),
            iaa.Sequential([
                iaa.imgcorruptlike.Brightness(severity=1),
                iaa.LinearContrast((0.6, 1.4)),
                iaa.Affine(
                    translate_percent={"x": (-0.25, 0.25)}
                )
            ]),
            iaa.Sequential([
                iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=0.5),
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                )
            ]),
            iaa.Sequential([
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                iaa.imgcorruptlike.GaussianNoise(severity=1),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
                )
            ]),
            iaa.Identity(),
            iaa.Identity()
        ]) # apply augmenters in random order

    def __call__(self, sample):
        new_images=np.array([np.array(sample) for _ in range(self.num_of_augms)])
        new_samples = self.seq(images = new_images)
        # new_samples=np.concatenate((new_samples, np.array([np.array(sample)])), axis=0)
        tensor_samples=torch.stack([F.normalize(F.to_tensor(s),(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) for s in new_samples])
        return tensor_samples
        

def my_collate(batch):
    target_rep = (batch[0][0].shape[0]) 
    data = torch.cat([item[0] for item in batch],dim=0)
    target = [item[1] for item in batch for _ in range(target_rep)]
    target = torch.LongTensor(target)
    return (data, target)

class RandomGrayScale(object):
    def __init__(self, p=0.2):
        self.p=p

    def __call__(self, sample):
        if torch.rand(1)<self.p:
            return F.to_grayscale(sample,num_output_channels=3)
        else:
            return sample


class SimpleAugment(object):
    def __init__(self, num_augments=1):
        self.num_augments = num_augments
        self.transform_anchor=transforms.Compose([
            RandomGrayScale(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_copy=transforms.Compose(
                        [transforms.RandomCrop(32, padding=4), # fill parameter needs torchvision installed from source
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
			             transforms.ToTensor(), 
                         Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])




    def __call__(self, sample):
        
        sample1=self.transform_anchor(sample)
        if self.num_augments==1:
            return torch.stack([sample1])
        else:
            transform_list=[sample1]
            for k in range(self.num_augments-1):
                transform_list.append(self.transform_copy(sample))


            return torch.stack(transform_list)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img