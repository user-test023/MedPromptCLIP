import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
import clip
from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform
import json
from templates.ODIR_templates import IMAGENET_TEMPLATES
from templates.mapper_data import ctx_templates


def build_data_loader(
        cfg,
        sampler_type="RandomSampler",
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None,
        class_names=None
):
    # Make Data Loader for Text-Only data
    if is_train and cfg.TRAINER.NAME == "MedPromptCLIP":
        data_source = DatasetWrapper_TextOnly(cfg, class_names)
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    if is_train and cfg.TRAINER.NAME == "MedPromptCLIP":
        # Build data loader for text dataset only!!!
        data_loader = torch.utils.data.DataLoader(
            data_source,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )
    else:
        # Build data loader for standard image-data
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
            self,
            cfg,
            custom_tfm_train=None,
            custom_tfm_test=None,
            dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
            class_names=dataset.classnames
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class DatasetWrapper_TextOnly(TorchDataset):
    def __init__(self, cfg, class_names):
        self.cfg = cfg
        self.class_names = class_names.copy()
        name_to_id = {}
        print(f'class:{self.class_names}')
        for idd, class_name in enumerate(self.class_names):
            name_to_id[class_name.lower().replace("_", " ")] = idd

        # add category
        if cfg.TRAINER.MEDPROMPTCLIP.EYE_DOMAIN and cfg.TRAINER.MEDPROMPTCLIP.GPT_PATH != "":
            file = open(cfg.TRAINER.MEDPROMPTCLIP.GPT_PATH, "r")
            GPT_prompt_dict = json.load(file)
            for single_key in GPT_prompt_dict.keys():
                formatted_key = single_key.replace("_", " ").lower()
                if formatted_key not in name_to_id:
                    new_id = len(name_to_id)
                    name_to_id[formatted_key] = new_id
                    self.class_names.append(single_key)

        # init
        self.text_input_list = []
        self.text_label_list = []
        self.one_hot_labels = []
        nctx_learnable = cfg.TRAINER.MEDPROMPTCLIP.N_CTX_TEXT

        # use templates
        if cfg.TRAINER.MEDPROMPTCLIP.USE_TEMPLATES:
            print(f"Using standard templates for data creation")
            for idx, class_name in enumerate(self.class_names):
                input_text = [t.format(class_name.replace("_", " ")) for t in IMAGENET_TEMPLATES]
                if nctx_learnable <= 5:
                    label_text = [input_text[0]] * (len(input_text) - 1)
                else:
                    prompt_prefix = " ".join(["X"] * nctx_learnable)
                    label_text = [prompt_prefix + " " + class_name.replace("_", " ") + "."] * (len(input_text) - 1)
                input_text = input_text[1:]
                labels = [name_to_id[class_name.lower().replace("_", " ")]] * len(input_text)
                self.text_input_list += input_text
                self.text_label_list += label_text
                self.one_hot_labels += labels

        # use attribute data
        if cfg.TRAINER.MEDPROMPTCLIP.USE_ATTRIBUTE_DATA:
            print(f"Using attribute prompt templates for data creation")
            for idx, class_name in enumerate(self.class_names):
                input_text = [t.format(class_name.replace("_", " ")) for t in ctx_templates]
                if nctx_learnable <= 5:
                    label_text = [input_text[0]] * (len(input_text) - 1)
                else:
                    prompt_prefix = " ".join(["X"] * nctx_learnable)
                    label_text = [prompt_prefix + " " + class_name.replace("_", " ") + "."] * (len(input_text) - 1)
                input_text = input_text[1:]
                labels = [name_to_id[class_name.lower().replace("_", " ")]] * len(input_text)
                self.text_input_list += input_text
                self.text_label_list += label_text
                self.one_hot_labels += labels

        # add GPT data
        if cfg.TRAINER.MEDPROMPTCLIP.GPT_PATH != "":
            file = open(cfg.TRAINER.MEDPROMPTCLIP.GPT_PATH, "r")
            GPT_prompt_dict = json.load(file)
            k = 0
            for single_key in GPT_prompt_dict.keys():
                single_key_formatted = single_key.replace("_", " ").lower()
                temp_input_text = GPT_prompt_dict[single_key]
                if nctx_learnable <= 5:
                    temp_label_text = ["a photo of a {}.".format(single_key_formatted)] * len(temp_input_text)
                else:
                    prompt_prefix = " ".join(["X"] * nctx_learnable)
                    temp_label_text = [prompt_prefix + " " + single_key + "."] * len(temp_input_text)
                # if not in class_names,it should be added to name_to_id
                if single_key_formatted in name_to_id:
                    temp_labels = [name_to_id[single_key_formatted]] * len(temp_input_text)
                    self.text_input_list += temp_input_text
                    self.text_label_list += temp_label_text
                    self.one_hot_labels += temp_labels
                    k += 1
            print(f"Total classes used from GPT EYECUPL dataset are {k}")

        assert len(self.text_input_list) == len(self.one_hot_labels) == len(self.text_label_list)
        print(f"Total number of text samples in the dataset are {len(self.text_input_list)}")
        self.data_source = self.text_input_list

    def __len__(self):
        return len(self.text_input_list)

    def __getitem__(self, idx):
        single_input_text = clip.tokenize(self.text_input_list[idx])#llm data
        single_input_label = clip.tokenize(self.text_label_list[idx])#label
        label = self.one_hot_labels[idx]
        output = {"input_text": single_input_label, "output_text": single_input_text, "label": label}
        return output