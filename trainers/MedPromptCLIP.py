import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .zsclip_cupl import CUSTOM_TEMPLATES_MEDPROMPT
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):#load the model according to cfg.backbone.name
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    if backbone_name == "RN50_PMC" or backbone_name == "ViT-L/14_PMC" or backbone_name == "ViT-L/14_PMC_2": 
        model_path = url
    elif backbone_name == "Ret" or backbone_name == "RN50_flair" or backbone_name == "ViT-L/14_unimed" or backbone_name =="ViLReF" or backbone_name == "RET":
        model_path = url
    elif backbone_name == "unimed" :
        model_path = url
    else:
        model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu",weights_only=True)
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,  # We are not doing any visual branch training
                          "language_depth": cfg.TRAINER.MEDPROMPTCLIP.PROMPT_DEPTH_TEXT,#add the prompt depth    learnable layer number
                          "vision_ctx": cfg.TRAINER.MEDPROMPTCLIP.N_CTX_VISION,   #transformer added 
                          "language_ctx": cfg.TRAINER.MEDPROMPTCLIP.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):#transform the prompts to text features
        """
        (batch, sequence, dim) → (sequence, batch, dim)
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):#for learn prompt
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.MEDPROMPTCLIP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                            "\nPlease use VPT trainer if you want to learn only vision " \
                                                            "branch"
        assert cfg.TRAINER.MEDPROMPTCLIP.PROMPT_DEPTH_VISION == 0, "MEDPROMPTCLIP only adapts language encoder during training"
        n_ctx = cfg.TRAINER.MEDPROMPTCLIP.N_CTX_TEXT#ctx number
        ctx_init = cfg.TRAINER.MEDPROMPTCLIP.CTX_INIT#init prompt
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]#input language embedding dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"#image need fit the clip model

        if ctx_init and n_ctx <= 5:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)#high level embedding into low level embedding
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]#get the embedding of the prompt a {modal} image of
            if cfg.TRAINER.MEDPROMPTCLIP.CROSS_DATASET:#if cross dataset, use the custom prompt
                prompt = CUSTOM_TEMPLATES_MEDPROMPT[cfg.DATASET.NAME]#use different prompt for different dataset
            else:
                prompt = "a photo of a {}."#else use its own prompt
            prompts = [prompt.format(c.replace("_", " ")) for c in classnames]
            prompt_prefix = "a photo of a"
        else:
            # random initialization ntx>4 for meaningful context
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)#XXXX
            prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        print(f"MedPromptCLIP design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.MEDPROMPTCLIP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)
        # Use the below variable only for inference.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        self.clip_model_zs = load_clip_to_cpu(cfg, True).float().cuda()

        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix):#for building the prompts
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,#sequence 
        )

        return prompts

    def forward(self, tokenized_prompts):#generate the prompts embedding
        ctx = self.ctx.float()
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(tokenized_prompts), -1, -1)

        # Adaptively create embedding using given input tokens
        embedding = self.clip_model_zs.token_embedding(tokenized_prompts)
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]#name
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):#for combine prompt with TextEncoder
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts.cuda()  # use only in inference
        self.text_encoder = TextEncoder(clip_model).float()   # 文本编码器：用于将提示和文本编码为特征向量
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, inputs, outputs=None):
        if self.prompt_learner.training:
            tokenized_texts_labels = outputs#llm data
            tokenized_texts_inputs = inputs#label
            # Calculate features
            with torch.no_grad():
                # encode target "a fundus image of CLS." meaningful word
                target_embed = self.prompt_learner.clip_model_zs.encode_text(tokenized_texts_labels)#use frozen clip to develop target_embed
                target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True)  # take its norm

            # encode inputs "a fundus image of CLS with some context" add prompt(prompt full of meaningful word)
            prompts = self.prompt_learner(tokenized_texts_inputs)  # encode input prompts     with n_ctx   生成与输入文本匹配的可学习提示向量（如形状[batch, n_ctx, dim]）
            text_features = self.text_encoder(prompts, tokenized_texts_inputs)#输入的提示和标记化文本add prompt to textencoder     prompt+label
            outputs = text_features / text_features.norm(dim=-1, keepdim=True)
            return outputs, target_embed#outputs is with prompt, target_embed is without prompt
        #test mode
        else:
            # Need to do zeroshot inference
            images = inputs
            with torch.no_grad():
                # encode test images
                image_features = self.prompt_learner.clip_model_zs.encode_image(images)#use frozen clip to develop image_features   防止训练prompt时改变模型的参数
                # take its norm
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Now make the prompts
                prompted_embedding = self.prompt_learner(self.tokenized_prompts)#use learned prompt to develop prompted_embedding
                # encode text features: test set class names
                text_features = self.text_encoder(prompted_embedding, self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()                # Compute the logits
                logits = logit_scale * image_features @ text_features.t()#calculate logits
                return logits


@TRAINER_REGISTRY.register()
class MedPromptCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MEDPROMPTCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MEDPROMPTCLIP.PREC == "fp32" or cfg.TRAINER.MEDPROMPTCLIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model.float())

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "clip_model_zs" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MEDPROMPTCLIP.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MEDPROMPTCLIP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output_inputs, output_targets = model(image, label)

            if self.cfg.TRAINER.MEDPROMPTCLIP.L_TWO_NORM:
                loss_ftn = torch.nn.MSELoss()
                loss = loss_ftn(output_inputs, output_targets) * \
                       self.cfg.TRAINER.MEDPROMPTCLIP.L_TWO_WEIGHT
            else:
                loss = F.l1_loss(output_inputs, output_targets, reduction='mean')

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):#load the batch data for training
        input = batch["input_text"]#label
        label = batch["output_text"]#llm data
        input = input.to(self.device).squeeze(1)
        label = label.to(self.device).squeeze(1)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"
        if epoch is None:
            epoch = self.cfg.OPTIM.MAX_EPOCH

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
