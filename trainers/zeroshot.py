import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
import json
from .coop import load_clip_to_cpu
# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from templates.imagenet_templates import IMAGENET_TEMPLATES
from templates.mapper_data import ctx_templates

CUSTOM_TEMPLATES_MEDPROMPT = {
    "Caltech101": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "modaldiff": "a photo of a {}.",
    "ODIR_3x200": "a fundus image of {}.",
    "ODIR": "a fundus image of {}.",
    "Ultrasound": "a Ultrasound retinal image of {}.",
    "oct_c8": "an oct retinal image of {}.",
    "SLID_E": "a Slit Lamp image of {}.",
    "OCTDL": "an oct retinal image of {}.",
    "f1000images": "a fundus image of {}.",
    "SLO": "a SLO image of {}.",
    "FFA": "a FFA image of {}."
}

CUSTOM_TEMPLATES = {
    "Caltech101": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "modaldiff": "a photo of a {}.",
    "ODIR_3x200": "a fundus image of {}.",
    "ODIR": "a fundus image of {}.",
    "Ultrasound": "a Ultrasound retinal image of {}.",
    "oct_c8": "an oct retinal image of {}.",
    "SLID_E": "a Slit Lamp image of {}.",
    "OCTDL": "an oct retinal image of {}.",
    "SLO": "a SLO image of {}.",
    "FFA": "a FFA image of {}."
}


@TRAINER_REGISTRY.register()
class Zeroshot(TrainerX):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES#模板初始化

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames#类信息

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)#冻结参数

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]#当前使用的数据集不是 "ImageNet"，则向模板中添加与当前数据集相对应的自定义模板

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):#遍历所有模板
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]#类的提示
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)#对提示进行分词
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
