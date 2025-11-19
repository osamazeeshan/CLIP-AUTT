import os
import sys
from os.path import dirname, abspath, join
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import torch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from  dlib.configure import constants
from dlib.fer_models.apvit import PoolingVitClassifier


__all__ = ['FerModel']

_IMAGENET = {
    constants.RESNET18: 'checkpoints/resnet18-f37072fd.pth',
    constants.RESNET34: 'checkpoints/resnet34-b627a593.pth',
    constants.RESNET50: 'checkpoints/resnet50-11ad3fa6.pth',
    constants.RESNET101: 'checkpoints/resnet101-cd907fc2.pth',
    constants.RESNET152: 'checkpoints/resnet152-f82ba261.pth',
}

_DEFAULT_APVIT = {
    "k": 160,
    "r": 0.9,
    "dense_dims": 'None',
    "attn_method": constants.ATT_SUM_ABS_1,
    "normalize_att": False,
    "apply_self_att": False,
    "hid_att_dim": 128,
    "pretrained": None,
    "freeze_backbone": False
}


class FerModel(nn.Module):
    def __init__(self, ncls: int, model_type: str, pretrained: bool,
                 apvit_config: dict):
        super(FerModel, self).__init__()

        dir_pre_w = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)
        torch.hub.set_dir(dir_pre_w)
        self.ncls = ncls
        self.model_type = model_type
        self.num_ftrs = 0

        if model_type.startswith('resnet'):
            self.model = torch.hub.load('pytorch/vision', model_type,
                                        weights=None, force_reload=False)

            if pretrained:
                cpu_dev = torch.device("cpu")
                path_w = join(dir_pre_w, _IMAGENET[model_type])
                w = torch.load(path_w, map_location=cpu_dev)
                self.model.load_state_dict(w, strict=True)

            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(num_ftrs, ncls)
            )

        elif model_type == constants.APVIT:
            config = apvit_config
            self.model = PoolingVitClassifier(num_classes=ncls, **config)

        else:
            raise NotImplementedError(model_type)

        # self.compound_cl_head = None
        # self.basic_ncls = ncls

    def create_compound_cl_head(self, ncls_c: int, dense_dims: str = 'None'):
        if self.model_type.startswith('resnet'):
            assert dense_dims == 'None', dense_dims
            self.compound_cl_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.num_ftrs, ncls_c)
            )

        elif self.model_type == constants.APVIT:
            self.compound_cl_head = self.model.create_compound_cl_head(
                ncls_c, dense_dims)

        else:
            raise NotImplementedError(self.model_type)

    def re_create_cl_head(self, ncls: int):
        assert self.compound_cl_head is None, "You cant do this case."

        if self.model_type.startswith('resnet'):
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.num_ftrs, ncls)
            )

        elif self.model_type == constants.APVIT:
            self.model.re_create_cl_head(ncls)

        else:
            raise NotImplementedError(self.model_type)

    def flush(self):
        if hasattr(self.model, 'flush'):
            self.model.flush()

    @staticmethod
    def freeze_modules(modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def freeze_classifier_head(self):
        if self.model_type.startswith('resnet'):
            modules = self.classifier.mudules()

        elif self.model_type == constants.APVIT:
            modules = self.model.classification_head.modules()

        else:
            raise NotImplementedError

        self.freeze_modules(modules)

        if self.compound_cl_head is not None:
            modules = self.compound_cl_head.modules()
            self.freeze_modules(modules)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:

        if self.model_type.startswith('resnet'):
            ft = self.model(x)
            logits = self.classifier(ft)

        elif self.model_type == constants.APVIT:
            ft = self.model(x)
            logits = self.model.classification_head(ft)

        else:
            raise NotImplementedError

        # if self.compound_cl_head is not None:
        #     logits = self.compound_cl_head(ft)

        return ft, logits

    def classify_ft(self, ft: torch.Tensor = None):
        # returns logits.
        if self.model_type.startswith('resnet'):
            logits = self.classifier(ft)

        elif self.model_type == constants.APVIT:
            logits = self.model.classification_head(ft)
        else:
            raise NotImplementedError

        return logits

    def predict_classes(self, logits: torch.Tensor):
        pass


def pre_load_all_resnet_imagenet():
    resnets = [constants.RESNET18, constants.RESNET34, constants.RESNET50,
               constants.RESNET101, constants.RESNET152]
    holder = {
        constants.RESNET18: 'ResNet18_Weights.DEFAULT',
        constants.RESNET34: 'ResNet34_Weights.DEFAULT',
        constants.RESNET50: 'ResNet50_Weights.DEFAULT',
        constants.RESNET101: 'ResNet101_Weights.DEFAULT',
        constants.RESNET152: 'ResNet152_Weights.DEFAULT',
    }
    destination = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)
    os.makedirs(destination, exist_ok=True)
    torch.hub.set_dir(destination)

    for md in resnets:
        torch.hub.load('pytorch/vision', md, weights=holder[md],
                       force_reload=False)


def run_resnet():
    resnets = [constants.RESNET18, constants.RESNET34, constants.RESNET50,
               constants.RESNET101, constants.RESNET152]
    ncls = 7
    cuda_dev = torch.device("cuda:1")
    x = torch.rand((32, 3, 224, 224), device=cuda_dev)

    for md in resnets:
        model = FerModel(ncls=ncls, model_type=md, pretrained=True,
                         apvit_config=None)
        model = model.to(cuda_dev)
        out = model(x)
        print(md, x.shape, out.shape)


def run_apvit_core():
    ncls = 7
    cuda_dev = torch.device("cuda:1")
    x = torch.rand((32, 3, 112, 112), device=cuda_dev)
    model = PoolingVitClassifier(num_classes=ncls, **_DEFAULT_APVIT).to(
        cuda_dev)
    out = model(x)
    print(x.shape, out.shape)


def run_apvit():
    ncls = 2
    cuda_dev = torch.device("cuda:3")
    x = torch.rand((2, 3, 112, 112), device=cuda_dev)

    model_type = constants.APVIT
    model = FerModel(ncls=ncls, model_type=model_type, pretrained=True,
                     apvit_config=_DEFAULT_APVIT)

    # model.create_compound_cl_head(ncls_c=11)

    model.to(cuda_dev)
    ft, logits = model(x)
    print(model_type, ft.shape, logits.shape)

    print(f"Compound logits shape: {logits.shape}")


if __name__ == "__main__":
    # pre_load_all_resnet_imagenet()
    # run_resnet()
    # run_apvit_core()
    run_apvit()
