import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
    # with open(image_list) as f:
    image_index = [x.split(' ')[0] for x in image_list]
    # with open(image_list) as f:
    label_list = []
    selected_list = []
    for ind, x in enumerate(image_list):
        label = x.split(' ')[1].strip()
        label_list.append(int(label))
        selected_list.append(ind)
    image_index = np.array(image_index)
    label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.return_index = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            if self.return_index:
                return img, target,index
            else:
                return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class STL(object):
    def __init__(self,  root="./data/multi/", ttype="labeled",
                 transform=None, target_transform=None, test=False):
        if ttype=="labeled":
            imgs=np.load(os.path.join(root,"labeled_data"))
            lables=np.load(os.path.join(root,"labeled_label"))
        elif ttype=="unlabeled":
            imgs=np.load(os.path.join(root,"unlabeled_data"))
            lables=np.load(os.path.join(root,"unlabeled_label"))
        else:
            imgs=np.load(os.path.join(root,"test_data"))
            lables=np.load(os.path.join(root,"test_label"))

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.return_index=False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        # path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = Image.fromarray(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            if self.return_index:
                return img, target,index
            else:
                return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)