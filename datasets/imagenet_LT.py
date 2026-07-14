import os
import pickle
import numpy as np
from collections import OrderedDict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import listdir_nohidden, mkdir_if_missing
from collections import Counter



def calculate_class_proportions(traindata_cls_counts):
    client_class_proportions = {}

    for client_id, class_counts in traindata_cls_counts.items():
        total_samples = sum(class_counts.values())
        proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        client_class_proportions[client_id] = proportions

    return client_class_proportions
def normalize_path(path):
    """Normalize path to use forward slashes."""
    return path.replace('\\', '/')

def get_class_distribution(train_data):

    labels = [item.label for item in train_data]

    class_distribution = Counter(labels)

    sorted_distribution = dict(sorted(class_distribution.items()))

    return sorted_distribution

# @DATASET_REGISTRY.register()
class ImageNet_LT(DatasetBase):
    loader_dir = "ImageNet_LT"
    dataset_dir = "/home/hsh/dataset/imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.loader_dir = os.path.join(root, self.loader_dir)
        self.image_dir_train = os.path.join(self.dataset_dir, 'train')
        self.image_dir_test = os.path.join(self.dataset_dir)
        # self.class_distribution = self.get_class_distribution()
        #
        # print("class_distribution",self.class_distribution)

        text_file = os.path.join(self.loader_dir, "classnames.txt")


        classnames = self.read_classnames(text_file)
        train = self.read_train_data(classnames, "ImageNet_LT_train.txt")
        test, label_to_classname = self.read_test_data(classnames, "ImageNet_LT_test.txt")

        class_distribution = get_class_distribution(train)
        print("class_distribution:",class_distribution)
        federated_train_x, traindata_cls_counts = self.generate_federated_dataset_imagenet(train, num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, beta=cfg.DATASET.BETA)
        client_proportions = calculate_class_proportions(traindata_cls_counts)

        self.y_train = np.array([item.label for item in train])
        self.data_test = test
        self.client_proportions = client_proportions
        super().__init__(train_x=train, federated_train_x=federated_train_x, test=test)

    @staticmethod
    def read_classnames(text_file):
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_train_data(self, classnames, split_file):
        split_file = os.path.join(self.loader_dir, split_file)
        items = []

        # Read the split file to get the mapping of image paths to labels
        image_to_label = {}
        with open(split_file, "r") as f:
            for line in f:
                impath, label = line.strip().split()
                image_to_label[impath] = int(label)

        # Get all folders (classes) in the image directory
        folders = sorted(f.name for f in os.scandir(self.image_dir_train) if f.is_dir())


        for folder in folders:
            classname = classnames[folder]
            imnames = listdir_nohidden(os.path.join(self.image_dir_train, folder))

            for imname in imnames:
                relative_impath = normalize_path(os.path.join('train', folder, imname))
                full_impath = os.path.join(self.image_dir_train, folder, imname)

                if relative_impath in image_to_label:
                    label = image_to_label[relative_impath]
                    item = Datum(impath=full_impath, label=label, classname=classname)
                    items.append(item)

        return items

    def read_test_data(self, classnames, split_file):
        split_file = os.path.join(self.loader_dir, split_file)
        items = []

        # Create a mapping from label numbers to class names
        label_to_classname = {i: name for i, name in enumerate(classnames.values())}

        with open(split_file, "r") as f:
            for line in f:
                file_path, label = line.strip().split()
                label = int(label)

                # Adjust the file path to match the actual structure
                # Remove the 'nXXXXXXXX/' part from the path
                adjusted_path = '/'.join(file_path.split('/')[0::2])

                # Get the full path to the image
                full_impath = os.path.join(self.image_dir_test, adjusted_path)

                # Get the classname using the label
                classname = label_to_classname[label]

                # Create a Datum object only if the file exists
                if os.path.exists(full_impath):
                    item = Datum(impath=full_impath, label=label, classname=classname)
                    items.append(item)
                else:
                    print(f"Warning: File not found: {full_impath}")

        return items, label_to_classname




