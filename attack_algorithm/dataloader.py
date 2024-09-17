import os.path
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import *
from PIL import Image
import urllib.request
import zipfile
from torch.utils.data import random_split
import random
root_map = {
    'cifar10': '.data/',
    'cifar100': '.data/',
    'subcifar': '.data/',
    'gtsrb': '.data/',
    'subgtsrb': '.data/',
    'svhn': '.data/',
    'flower': '.data/',
    'pubfig': '../dataset/pubfig',
    'eurosat': '.data/eurosat',
    'imagenet': '../imagenet/',
    'imagenet64': '../imagenet64/',
    'lisa': '.data/',
    'mnist': '.data/',
    'mnistm': '.data/',
    'food': '.data/',
    'pet': '.data/',
    'resisc': '.data/NWPU-RESISC45/',
}
mean_map = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.4914, 0.4822, 0.4465),
    'subcifar': (0.4914, 0.4822, 0.4465),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'imagenet64': (0.485, 0.456, 0.406),
    'flower': (0.485, 0.456, 0.406),
    'caltech101': (0.485, 0.456, 0.406),
    'stl10': (0.485, 0.456, 0.406),
    'iris': (0.485, 0.456, 0.406),
    'fmnist': (0.2860, 0.2860, 0.2860),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.3403, 0.3121, 0.3214),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.3403, 0.3121, 0.3214),#
    'pubfig': (129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0),
    'lisa': (0.3403, 0.3121, 0.3214),  #(0.4563, 0.4076, 0.3895),
    'eurosat': (0.3442, 0.3802, 0.4077),
     'food': (0.485, 0.456, 0.406),
    'pet': (0.485, 0.456, 0.406),
    'resisc': (0.485, 0.456, 0.406),
    'unknown': (0.5, 0.5, 0.5),
}
std_map = {
    'cifar10': (0.2023, 0.1994, 0.201),
    'cifar100': (0.2023, 0.1994, 0.201),
    'subcifar': (0.2023, 0.1994, 0.201),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'imagenet64': (0.229, 0.224, 0.225),
    'flower': (0.229, 0.224, 0.225),
    'caltech101': (0.229, 0.224, 0.225),
    'stl10': (0.229, 0.224, 0.225),
    'iris': (0.229, 0.224, 0.225),
    'fmnist': (0.3530, 0.3530, 0.3530),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.2724, 0.2608, 0.2669),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.2724, 0.2608, 0.2669),#
    'pubfig': (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),  # (1.0, 1.0, 1.0), #
    'lisa': (0.2724, 0.2608, 0.2669),  #(0.2298, 0.2144, 0.2259),
    'eurosat': (0.2036, 0.1366, 0.1148),
     'food': (0.229, 0.224, 0.225),
     'pet': (0.229, 0.224, 0.225),
    'resisc': (0.229, 0.224, 0.225),
    'unknown': (0.229, 0.224, 0.225),
}
num_class_map = {
    'cifar10': 10,
    'cifar100': 100,
    'subcifar': 2,
    'subgtsrb': 2,
    'svhn': 10,
    'pubfig': 83,
    'gtsrb': 43,
    'eurosat': 10,
    'flower': 102,
    'imagenet': 1000,
    'imagenet64': 1000,
    'mnist': 10,
    'mnistm': 10,
    'food': 101,
    'pet': 37,
    'resisc': 45,
}

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, transform=None, download=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        if download:
            self.download()
        class_path = os.path.join(root_dir, 'GTSRB/Final_Training/Images')
        for class_id in os.listdir(class_path):
            class_dir = os.path.join(class_path, class_id)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.ppm'):
                        self.images.append(os.path.join(class_dir, image_name))
                        self.labels.append(int(class_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]

        ppm_image = Image.open(img_path)
        rgb_image = ppm_image.convert('RGB')

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label

    def download(self):
        dataset_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        zip_filename = os.path.join(self.root_dir, "GTSRB_Final_Training_Images.zip")

        # Download the dataset
        if os.path.exists(zip_filename):
            pass # print("zip file is here, skip download")
        else:
            urllib.request.urlretrieve(dataset_url, zip_filename)

        # Extract the dataset
        if os.path.exists(os.path.join(self.root_dir, "GTSRB")):
            pass # print("GTSRB directory is here, skip unzip")
        else:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        # print("GTSRB dataset is ready.")

def CIFAR10Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    dataset = CIFAR10(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def CIFAR100Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    print(os.path.join(os.getcwd(), root))
    dataset = CIFAR100(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def GTSRBLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    g = torch.Generator()
    g.manual_seed(0)

    set_seed(0)
    dataset = GTSRBDataset(root, transform, download=True)
    ppm_samples_count = len(dataset)
    train_size = 33200
    train_dataset, test_dataset = random_split(dataset, [train_size, ppm_samples_count - train_size])
    # dataset = GTSRB(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    if split == 'train':
        dataset = train_dataset
    else:
        dataset = test_dataset
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=seed_worker, # deterministic running
        generator=g,
    )

def EuroSatLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    root = os.path.join(root, split)
    dataset = ImageFolder(root, transform)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def SVHNLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = SVHN(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )


loader_map = {
    'cifar10': CIFAR10Loader,
    'cifar100': CIFAR100Loader,
    'gtsrb': GTSRBLoader,
    'svhn': SVHNLoader,
    'eurosat': EuroSatLoader,
}


class AllLoader:
    def __init__(self, task, **kwargs):
        self.kwargs = kwargs
        self.task = task
        self.image_number = kwargs['image_number']
        # self.use_all_data_for_search = True #if self.task != 'cifar100' else False
        self.use_all_data_for_search = True
        self.consist_3_state = True
        self.device = kwargs['device']
        self.target_class = kwargs['target_class']
        self.num_class = num_class_map[task]
        self.image_size = kwargs['image_size']
        self.seed = 0 #np.random.randint(0, 1000)
        # print(f"all loader seed: {self.seed}")

        self.train_batch_size = kwargs['train_batch_size']
        self.test_batch_size = kwargs['test_batch_size']

        self.image_mean = mean_map[self.task]
        self.image_std = std_map[self.task]
        self.transform = {}
        self.domain_shift = kwargs['domain_shift']
        self.deterministic_run = kwargs['deterministic_run']
        self.deepvenom_transform = kwargs['deepvenom_transform'] if 'deepvenom_transform' in kwargs.keys() else 'normal'

        print(f"image loader initialized")

    def init_loader(self, pure=False):# when pure set to True, only initialize train and test loader

        self.percent = self.kwargs['attacker_data_percent']
        self.limited_image_mode = True if self.kwargs['limited_image_mode'] == 'yes' else False
        self.attacker_image_number = self.kwargs['attacker_image_number']

        if self.domain_shift != 0:
            if self.task == 'gtsrb':
                self.task = 'lisa'
                self.target_class = 40
                self.num_class = 47
            elif self.task == 'svhn':
                self.task = 'mnist'

        self.init_transform()

        self.train_loader = loader_map[self.task](root_map[self.task], batch_size=self.train_batch_size, split='train',
                                                  transform=self.transform['train'])
        self.test_loader = loader_map[self.task](root_map[self.task], batch_size=self.test_batch_size, split='test',
                                                 transform=self.transform['test'])
        self.total_number_of_train_images = self.train_loader.dataset.__len__()

        if not pure:
            if not self.deterministic_run:
                # trigger generation dataset is the same as attacker train loader
                self.attacker_train_loader, self.attacker_test_loader = self.AttackLoader(self.train_loader, self.test_loader,
                                                                                          self.num_class,
                                                                                          self.percent)
                # neuron selection : dataset_target & others; bit search: bit search data loader
                self.bit_search_data_loader, self.dataset_target, self.dataset_others = self.init_target_other_dataloader()  # self.attacker_train_dataset,

    def init_transform(self):
        size = (self.image_size, self.image_size)
        normalize = transforms.Normalize(mean_map[self.task], std_map[self.task])

        self.transform['test'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,
        ])
        transform_list = []
        transform_list.append(transforms.Compose([]))
        transform_list.append(transforms.RandomResizedCrop(self.image_size))
        if self.task in ['gtsrb', 'svhn', 'lisa', 'mnist']:
            transform_list.append(transforms.RandomRotation(10))
        elif self.task in ['pubfig', 'flower']:
            transform_list.append(transforms.RandomHorizontalFlip(1.))
            transform_list.append(transforms.RandomRotation(30))
        elif self.task in ['eurosat', 'cifar10', 'food', 'pet', 'resisc']:
            transform_list.append(transforms.RandomHorizontalFlip(1.))
            transform_list.append(transforms.RandomVerticalFlip(1.))
            transform_list.append(transforms.RandomRotation(30))
        if self.limited_image_mode and False:
            print(f"launch data augmentatin for limited image mode")
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            transform_list.append(transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)))
        self.random_choice = transforms.RandomChoice(transform_list)
        self.transform['train'] = transforms.Compose([
            self.random_choice,
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,
        ])
        if self.limited_image_mode and False:
            print(f"launch data augmentatin for limited image mode")
            self.transform['train'] = transforms.Compose([
                self.random_choice,
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Resize(size, antialias=None),
                normalize,
            ])

        self.transform['train_determ'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,
        ])
        if self.task == 'lisa':
            self.transform['test'] = transforms.Compose([
            transforms.Resize(size),
            normalize,
            ])
            self.transform['train'] = transforms.Compose([
                self.random_choice,
                transforms.Resize(size),
                normalize,
            ])
            self.transform['train_determ'] = transforms.Compose([
                transforms.Resize(size),
                normalize,
            ])
        elif self.task == 'mnist':
            self.transform['test'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Resize(size),
                normalize,
            ])
            self.transform['train'] = transforms.Compose([
                self.random_choice,
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Resize(size),
                normalize,
            ])
            self.transform['train_determ'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Resize(size),
                normalize,
            ])

        if self.deepvenom_transform == 'pure':
            print(f'using the simplest transform for verificaiton')
            t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,])
            self.transform['test'] = t
            self.transform['train'] = t
            self.transform['train_determ'] = t


    def init_transform_verify(self):
        # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        print(f"Reyad's transformation for normal fine-tuning")
        size = (self.image_size, self.image_size)
        normalize = transforms.Normalize(mean_map[self.task], std_map[self.task])
        self.transform['test'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            normalize,
        ])
        transform_list = []
        transform_list.append(transforms.Compose([]))
        transform_list.append(transforms.RandomResizedCrop(self.image_size))
        if self.task in ['gtsrb', 'svhn', 'lisa', 'mnist']:
            transform_list.append(transforms.RandomRotation(10))
        elif self.task in ['eurosat', 'cifar10', 'food', 'pet', 'resisc']:
            transform_list.append(transforms.RandomHorizontalFlip(1.))
            transform_list.append(transforms.RandomVerticalFlip(1.))
            transform_list.append(transforms.RandomRotation(30))

        self.random_choice = transforms.RandomChoice(transform_list)
        self.transform['train'] = transforms.Compose([
            transforms.Pad(4, padding_mode="constant"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

        self.transform['train_determ'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            normalize,
        ])

    def init_target_other_dataloader(self): # for selecting salient neuron (Avg feature map of target class - Avg feature map of other classes)
        determ_train_loader = loader_map[self.task](root_map[self.task], batch_size=self.train_batch_size,
                                                    split='train',
                                                    transform=self.transform['train_determ'])
        attacker_temp_train_loader, _ = self.AttackLoader(determ_train_loader, None, self.num_class, self.percent)

        if self.image_size == 224:
            batch_size = 32
        else:
            batch_size = 128

        if self.use_all_data_for_search:
            print(f"using all data for bit search")
            bit_search_data_loader = DataLoader(attacker_temp_train_loader.dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=2)
        else:
            # critical bit search dataloader: params select, bit search
            data_len = attacker_temp_train_loader.dataset.__len__()
            index = np.arange(data_len)
            np.random.seed(self.seed)
            np.random.shuffle(index)
            self.image_number = min(self.image_number, data_len)

            shuffle_dataset = Subset(attacker_temp_train_loader.dataset, index[:self.image_number])

            bit_search_data_loader = DataLoader(shuffle_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            for i in range(2):
                for input, label in bit_search_data_loader:
                    print(f'the label of first bit search data batch: {label[:10]}')
                    break

        if self.consist_3_state:
            dataset_target = self.get_target_subset(bit_search_data_loader.dataset, self.target_class)

            dataset_others = self.get_target_subset(bit_search_data_loader.dataset, self.target_class, anti=True)
        else:
            dataset_target = self.get_target_subset(attacker_temp_train_loader.dataset, self.target_class)

            dataset_others = self.get_target_subset(attacker_temp_train_loader.dataset, self.target_class, anti=True)

        print(f"compute the target and other dataset for attacker: {len(dataset_target)} & {len(dataset_others)}")

        return bit_search_data_loader, dataset_target, dataset_others

    def get_mean_std_fm(self, dataset, model, probabilistic=False, **kwargs):
        model.eval()
        fm_all = 0
        fm_list = []
        fm_probability = torch.zeros(1)
        use_trigger = False
        if 'use_trigger' in kwargs.keys():
            use_trigger = kwargs['use_trigger']
            sythesize_poison_image = kwargs['function']
            trigger = kwargs['trigger']

        data_loader = DataLoader(dataset, batch_size=self.train_loader.batch_size, shuffle=False,
                                 num_workers=0 if self.task != 'gtsrb' else 4, drop_last=False)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                # if i >= 0.1 * len(data_loader): break
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if use_trigger:
                    poison_inputs = sythesize_poison_image(inputs, trigger)
                    _, fm = model(poison_inputs, latent=True)
                else:
                    _, fm = model(inputs, latent=True)
                fm_sum = torch.sum(fm, dim=0) / inputs.size()[0]
                fm_all = torch.add(fm_sum.detach(), fm_all)
                # fm_all = torch.add(fm.detach(), fm_all)
                if probabilistic:
                    fm[fm > 0.0] = 1.0
                    fm[fm <= 0.0] = 0.0
                    fm_probability = torch.add(fm, fm_probability)

        if probabilistic:
            fm_probability = torch.div(fm_probability, len(data_loader))
            fm_probability = torch.sum(fm_probability, dim=0) / (data_loader.batch_size)
            fm_probability = torch.unsqueeze(fm_probability, dim=0)

        mean = torch.div(fm_all, len(data_loader))

        return mean.to(self.device), fm_probability.to(self.device)

    def get_target_subset(self, dataset, target=None, data_length=0, anti=False):
        assert isinstance(target, int)
        length = dataset.__len__() if data_length == 0 else data_length
        # index = get_index_subset(args, dataset)
        new_index = []
        for i in range(length):
            sample, label = dataset.__getitem__(i)
            if anti == False and target == label:
                new_index.append(i)
            elif anti == True and target != label:
                new_index.append(i)
            if len(new_index) >= length: break
        target_subset = torch.utils.data.Subset(dataset, new_index)
        return target_subset

    def AttackLoader(self, train_loader, test_loader, num_class, percent):
        if percent == 1.0 and self.limited_image_mode == False:
            return train_loader, test_loader
        total_len = train_loader.dataset.__len__()
        if self.limited_image_mode == True:
            part_len = self.attacker_image_number
        else:
            part_len = int(percent * total_len)
        index = np.arange(total_len)
        np.random.seed(self.seed)
        np.random.shuffle(index)

        if part_len // 128 != 0:
            part_len = int(part_len / 128) * 128 + 128

        train_length = part_len - 256
        test_length = 256
        #self.attacker_image_number if test_length < self.attacker_image_number else test_length
        print("attacker dataset: ", total_len, train_length, test_length)
        assert train_length > num_class

        # train_number = split_number_on_average(train_length, num_class)
        # test_number = split_number_on_average(test_length, num_class)
        # train_number_0 = [0] * num_class
        # test_number_0 = [0] * num_class
        # train_index = []
        # test_index = []

        # for idx in index:
        #     if len(train_index) >= train_length:
        #         break
        #     _, label = train_loader.dataset.__getitem__(idx)
        #     if train_number_0[label] < train_number[label]:
        #         train_number_0[label] += 1
        #         train_index.append(idx)
        # for idx in index:
        #     if idx in train_index: continue
        #     if len(test_index) >= test_length:
        #         break
        #     _, label = train_loader.dataset.__getitem__(idx)
        #     if test_number_0[label] < test_number[label]:
        #         test_number_0[label] += 1
        #         test_index.append(idx)

        train_index = index[:train_length]
        test_index = index[train_length: train_length + test_length]

        sub_train_dataset = Subset(train_loader.dataset, train_index)
        print(f'train length {len(train_index)} \ntrain index: {train_index[:10]}')
        sub_test_dataset = Subset(train_loader.dataset, test_index)
        attacker_train_loader = DataLoader(sub_train_dataset, batch_size=train_loader.batch_size, shuffle=True,
                                           num_workers=2, drop_last=True)
        if test_loader:
            attacker_test_loader = DataLoader(sub_test_dataset,
                                              batch_size=test_length if test_length < 2 * test_loader.batch_size else test_loader.batch_size,
                                              shuffle=False, num_workers=2,
                                              drop_last=True)
        else:
            attacker_test_loader = None

        return attacker_train_loader, attacker_test_loader


