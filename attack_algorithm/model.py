import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import numpy as np
import random


def set_seed(seed):
    if seed is None:
        random.seed()
        np.random.seed(None)
        torch.manual_seed(int(torch.initial_seed()))
        torch.cuda.manual_seed_all(int(torch.initial_seed()))
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class LatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = None
        self.feature_maps = []
        self.gradient = None
        self.num_ftrs = None
        self.neuron_score = 0.0
        self.last_layer = None
        self.suppress_neuron = None
        self.seed = 0

    def save_feature_map_in(self):
        def fn(_, input, output):
            self.feature_map = input[0]
        return fn

    def save_feature_map_out(self):
        def fn(_, input, output):
            self.feature_map = output.view(output.size()[0], -1)
        return fn

    def save_feature_map_outs(self):
        def fn(_, input, output):
            self.feature_maps.append(output.view(output.size()[0], -1))
        return fn

    def initial_layer(self, m):
        init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    def seed_initial(self, seed):
        if seed is None:
            random.seed()
            np.random.seed(None)
            torch.manual_seed(int(torch.initial_seed()))
            torch.cuda.manual_seed_all(int(torch.initial_seed()))
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def replace_last_layer(self, module, num_ftrs, num_class, replace, conv=False):
        if not replace: return module
        # print('warning, replacing with multi layers')
        if self.seed == 1:
            self.seed_initial(0)
        else:
            self.seed_initial(self.seed)
        if conv:
            module = nn.Conv2d(num_ftrs, num_class, kernel_size=(1, 1), stride=(1, 1))
        else:
            module = nn.Linear(num_ftrs, num_class)
            # module = nn.Sequential(
            #     nn.Linear(num_ftrs, 512),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(512, num_class)
            # )

        if self.seed == 1:
            module.weight.data = module.weight.data * -1.0 # seed 1 is set to reverse seed of seed 0.
        # print(f'seed: {self.seed}')#; last layer weights: {module.weight.data[0, :5]}')
        return module

    def register_multi_hook(self, name_list):
        for name, layer in self.model.named_modules():
            # print(name)
            if name in name_list:
                layer.__name__ = name
                layer.register_forward_hook(self.save_feature_map_outs())

    def forward(self, x, latent=False, multi_latent=False):
        self.feature_maps = []
        output = self.model(x)
        if multi_latent:
            return output, self.feature_maps
        if latent:
            return output, self.feature_map
        else:
            return output

class resnet18(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'resnet18'
        self.seed = seed
        self.model = models.resnet18(pretrained=pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.feature_maps = []
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend([ 'layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.fc = self.replace_last_layer(self.model.fc, self.num_ftrs, num_classes, replace)
        self.model.fc.register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.fc

class resnet50(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'resnet50'
        self.seed = seed
        self.model = models.resnet50(pretrained=pretrained)
        self.num_ftrs = self.model.fc.in_features
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend(['layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)
        self.model.fc = self.replace_last_layer(self.model.fc, self.num_ftrs, num_classes, replace)
        self.model.fc.register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.fc

class vgg16_bn(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'vgg16_bn'
        self.seed = seed
        self.model = models.vgg16_bn(pretrained=pretrained)
        self.num_ftrs = 25088
        self.model.classifier = self.replace_last_layer(self.model.classifier, self.num_ftrs, num_classes, replace)
        if multi_features:
            hook_names = ['features.' + str(ele) for ele in [16, 23, 30]] #4, 9,
            self.register_multi_hook(hook_names)

        self.hook = self.model.features.register_forward_hook(self.save_feature_map_out())
        self.last_layer = self.model.classifier

class vit(LatentModel):
    """
    Model used to pretrain.
    """

    def __init__(self, num_classes=1000, pretrained=True, replace=True, seed=None, multi_features=False):
        super().__init__()
        self.model_name = 'vit'
        self.seed = seed
        self.model = models.vit_b_16(pretrained=pretrained)
        self.num_ftrs = self.model.heads.head.in_features
        self.feature_maps = []
        if multi_features:
            hook_names = ['maxpool']
            hook_names.extend([ 'layer' + str(i) for i in range(1, 5)])
            self.register_multi_hook(hook_names)

        self.model.heads.head = self.replace_last_layer(self.model.heads.head, self.num_ftrs, num_classes, replace)
        self.model.heads.head.register_forward_hook(self.save_feature_map_in())
        self.last_layer = self.model.heads.head
        self.model.num_classes = num_classes

def map_model(**kwargs):
    model_name = kwargs['model_name']
    num_class = kwargs['num_class']
    pretrained = kwargs['pretrained']
    replace = kwargs['replace']
    seed = kwargs['seed'] if "seed" in kwargs.keys() else None
    if 'multi_features' not in kwargs.keys():
        multi_features = False
    else:
        multi_features = kwargs['multi_features']

    print(f"model initialized on seed {seed}")
    if model_name == 'resnet18':
        return resnet18(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'resnet50':
        return resnet50(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'vgg16_bn':
        return vgg16_bn(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])
    elif model_name == 'vit':
        return vit(num_class, pretrained, replace, seed, multi_features).to(kwargs['device'])



if __name__ == '__main__':
    pass