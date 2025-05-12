import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode
# 在导入部分添加
import cv2
import torchvision.transforms as T
from PIL import Image
from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch
#transform IMAGES
AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "instance_norm",
    "random_crop",
    "random_translation",
    "center_crop",  # This has become a default operation during testing
    "cutout",
    "imagenet_policy",
    "cifar10_policy",
    "svhn_policy",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "gaussian_noise",
    "colorjitter",
    "randomgrayscale",
    "gaussian_blur",
    "brightness_enhancement",  # 亮度增强
    "adaptive_binarization",   # 自适应二值化
    "otsu_binarization",       # Otsu二值化
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class BrightnessEnhancement:
    """
    增强图像亮度，适用于亮度过低的图像。
    
    参数:
        alpha (float): 对比度增强因子，大于1会增加对比度。
        beta (float): 亮度调整值，正值增加亮度，负值降低亮度。
        p (float): 应用此变换的概率。
    """
    
    def __init__(self, alpha=1.5, beta=30, p=0.5):
        self.alpha = alpha  # 对比度增强因子
        self.beta = beta    # 亮度调整值
        self.p = p          # 应用概率
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img  # 按概率决定是否应用此变换
            
        # 将PIL或Tensor图像转换为numpy数组以进行处理
        if isinstance(img, torch.Tensor):
            # 如果是Tensor格式，先转回PIL，再转numpy
            if img.dim() == 3 and img.shape[0] in [1, 3]:  # CHW格式
                # 将范围从[0,1]转到[0,255]
                np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                return img  # 不支持的Tensor格式
        else:
            # 假设是PIL图像
            np_img = np.array(img)
            
        # 应用亮度和对比度调整: new_img = alpha * old_img + beta
        adjusted = cv2.convertScaleAbs(np_img, alpha=self.alpha, beta=self.beta)
        
        # 转回原始格式
        if isinstance(img, torch.Tensor):
            # 转回CHW格式的Tensor，范围[0,1]
            result = torch.from_numpy(adjusted).float().permute(2, 0, 1) / 255.0
            return result
        else:
            # 转回PIL图像
            return Image.fromarray(adjusted)


class AdaptiveBinarization:
    """
    应用自适应二值化，更好地处理照明不均匀的图像。
    
    参数:
        block_size (int): 邻域大小，必须为奇数。
        c (int): 从平均值或加权平均值中减去的常数。
        p (float): 应用此变换的概率。
    """
    
    def __init__(self, block_size=11, c=2, p=0.5):
        # 确保block_size为奇数
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c
        self.p = p
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
            
        # 将图像转换为numpy数组以进行处理
        if isinstance(img, torch.Tensor):
            # 如果是Tensor，先转为numpy
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                return img
        else:
            # 假设是PIL图像
            np_img = np.array(img)
        
        # 如果是彩色图像，先转为灰度
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_img
            
        # 应用自适应二值化
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            self.block_size, 
            self.c
        )
        
        # 如果原图是彩色的，将二值化结果转回三通道
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # 转回原始格式
        if isinstance(img, torch.Tensor):
            result = torch.from_numpy(binary).float().permute(2, 0, 1) / 255.0
            return result
        else:
            return Image.fromarray(binary)


class OtsuBinarization:
    """
    应用Otsu二值化方法，自动确定最佳阈值。
    
    参数:
        p (float): 应用此变换的概率。
    """
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
            
        # 将图像转换为numpy数组以进行处理
        if isinstance(img, torch.Tensor):
            # 如果是Tensor，先转为numpy
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                return img
        else:
            # 假设是PIL图像
            np_img = np.array(img)
        
        # 如果是彩色图像，先转为灰度
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_img
            
        # 应用Otsu二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 如果原图是彩色的，将二值化结果转回三通道
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # 转回原始格式
        if isinstance(img, torch.Tensor):
            if len(binary.shape) == 3:
                result = torch.from_numpy(binary).float().permute(2, 0, 1) / 255.0
            else:
                result = torch.from_numpy(binary).float().unsqueeze(0) / 255.0
            return result
        else:
            return Image.fromarray(binary)

class Random2DTranslation:
    """Given an image of (height, width), we resize it to
    (height*1.125, width*1.125), and then perform random cropping.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``torchvision.transforms.functional.InterpolationMode.BILINEAR``
    """

    def __init__(
        self, height, width, p=0.5, interpolation=InterpolationMode.BILINEAR
    ):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img=img,
                size=[self.height, self.width],
                interpolation=self.interpolation
            )

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = F.resize(
            img=img,
            size=[new_height, new_width],
            interpolation=self.interpolation
        )
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = F.crop(
            img=resized_img,
            top=y1,
            left=x1,
            height=self.height,
            width=self.width
        )

        return croped_img


class InstanceNormalization:
    """Normalize data using per-channel mean and standard deviation.

    Reference:
        - Ulyanov et al. Instance normalization: The missing in- gredient
          for fast stylization. ArXiv 2016.
        - Shu et al. A DIRT-T Approach to Unsupervised Domain Adaptation.
          ICLR 2018.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img-mean) / (std + self.eps)


class Cutout:
    """Randomly mask out one or more patches from an image.

    https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes (int, optional): number of patches to cut out
            of each image. Default is 1.
        length (int, optinal): length (in pixels) of each square
            patch. Default is 16.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
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

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class GaussianNoise:
    """Add gaussian noise."""

    def __init__(self, mean=0, std=0.15, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise


def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, target_size, normalize)
    else:
        return _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        print(f"+ resize to {target_size}")
        tfm_train += [Resize(input_size, interpolation=interp_mode)]

    if "random_translation" in choices:
        print("+ random translation")
        tfm_train += [Random2DTranslation(input_size[0], input_size[1])]

    if "random_crop" in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print(f"+ random crop (padding = {crop_padding})")
        tfm_train += [RandomCrop(input_size, padding=crop_padding)]

    if "random_resized_crop" in choices:
        s_ = cfg.INPUT.RRCROP_SCALE
        print(f"+ random resized crop (size={input_size}, scale={s_})")
        tfm_train += [
            RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
        ]

    if "random_flip" in choices:
        print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]

    if "imagenet_policy" in choices:
        print("+ imagenet policy")
        tfm_train += [ImageNetPolicy()]

    if "cifar10_policy" in choices:
        print("+ cifar10 policy")
        tfm_train += [CIFAR10Policy()]

    if "svhn_policy" in choices:
        print("+ svhn policy")
        tfm_train += [SVHNPolicy()]

    if "randaugment" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        m_ = cfg.INPUT.RANDAUGMENT_M
        print(f"+ randaugment (n={n_}, m={m_})")
        tfm_train += [RandAugment(n_, m_)]

    if "randaugment_fixmatch" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment_fixmatch (n={n_})")
        tfm_train += [RandAugmentFixMatch(n_)]

    if "randaugment2" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment2 (n={n_})")
        tfm_train += [RandAugment2(n_)]

    if "colorjitter" in choices:
        b_ = cfg.INPUT.COLORJITTER_B
        c_ = cfg.INPUT.COLORJITTER_C
        s_ = cfg.INPUT.COLORJITTER_S
        h_ = cfg.INPUT.COLORJITTER_H
        print(
            f"+ color jitter (brightness={b_}, "
            f"contrast={c_}, saturation={s_}, hue={h_})"
        )
        tfm_train += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

    if "randomgrayscale" in choices:
        print("+ random gray scale")
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if "gaussian_blur" in choices:
        print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
        gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
        tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

    print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "cutout" in choices:
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN
        print(f"+ cutout (n_holes={cutout_n}, length={cutout_len})")
        tfm_train += [Cutout(cutout_n, cutout_len)]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_train += [normalize]

    if "gaussian_noise" in choices:
        print(
            f"+ gaussian noise (mean={cfg.INPUT.GN_MEAN}, std={cfg.INPUT.GN_STD})"
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]
    
    # 添加亮度增强选项
    if "brightness_enhancement" in choices:
        alpha = getattr(cfg.INPUT, "BRIGHTNESS_ALPHA", 1.5)  # 提取配置或使用默认值
        beta = getattr(cfg.INPUT, "BRIGHTNESS_BETA", 30)
        p = getattr(cfg.INPUT, "BRIGHTNESS_P", 0.5)
        print(f"+ brightness enhancement (alpha={alpha}, beta={beta}, p={p})")
        tfm_train += [BrightnessEnhancement(alpha=alpha, beta=beta, p=p)]
    
    # 添加自适应二值化选项
    if "adaptive_binarization" in choices:
        block_size = getattr(cfg.INPUT, "BINARIZE_BLOCK_SIZE", 11)
        c = getattr(cfg.INPUT, "BINARIZE_C", 2)
        p = getattr(cfg.INPUT, "BINARIZE_P", 0.5)
        print(f"+ adaptive binarization (block_size={block_size}, c={c}, p={p})")
        tfm_train += [AdaptiveBinarization(block_size=block_size, c=c, p=p)]
    
    # 添加Otsu二值化选项
    if "otsu_binarization" in choices:
        p = getattr(cfg.INPUT, "OTSU_P", 0.5)
        print(f"+ otsu binarization (p={p})")
        tfm_train += [OtsuBinarization(p=p)]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_train += [InstanceNormalization()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(input_size)]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_test += [normalize]
    
    # 添加亮度增强选项
    if "brightness_enhancement" in choices:
        alpha = getattr(cfg.INPUT, "BRIGHTNESS_ALPHA", 1.5)  # 提取配置或使用默认值
        beta = getattr(cfg.INPUT, "BRIGHTNESS_BETA", 30)
        p = getattr(cfg.INPUT, "BRIGHTNESS_P", 0.5)
        print(f"+ brightness enhancement (alpha={alpha}, beta={beta}, p={p})")
        tfm_test += [BrightnessEnhancement(alpha=alpha, beta=beta, p=p)]
    
    # 添加自适应二值化选项
    if "adaptive_binarization" in choices:
        block_size = getattr(cfg.INPUT, "BINARIZE_BLOCK_SIZE", 11)
        c = getattr(cfg.INPUT, "BINARIZE_C", 2)
        p = getattr(cfg.INPUT, "BINARIZE_P", 0.5)
        print(f"+ adaptive binarization (block_size={block_size}, c={c}, p={p})")
        tfm_test += [AdaptiveBinarization(block_size=block_size, c=c, p=p)]
    
    # 添加Otsu二值化选项
    if "otsu_binarization" in choices:
        p = getattr(cfg.INPUT, "OTSU_P", 0.5)
        print(f"+ otsu binarization (p={p})")
        tfm_test += [OtsuBinarization(p=p)]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test
