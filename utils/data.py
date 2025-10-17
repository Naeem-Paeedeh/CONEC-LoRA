import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import random
import json
from utils.class_names import domainnet_classnames, officehome_classnames
import csv


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iGanFake(iData):
    MANY_SHOT_THRES = 70
    FEW_SHOT_THRES = 20
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args.class_order
        self.class_order = class_order
        self.cls_num = 2

    def make_imb(self, dataset_dict, root_dir):
        imb_info = {
            "gaugan": [3000, 150],
            "biggan": [120, 1200],
            "wild": [155, 3115],
            "whichfaceisreal": [600, 120],
            "san": [130, 130],
        }
        for name in dataset_dict.keys():
            pos_trian_num = imb_info[name][0]
            neg_train_num = imb_info[name][1]
            pos_train = dataset_dict[name]["train_pos_naive"]
            neg_train = dataset_dict[name]["train_neg_naive"]
            random.shuffle(pos_train)
            pos_train = pos_train[:pos_trian_num]
            random.shuffle(neg_train)
            neg_train = neg_train[:neg_train_num]
            dataset_dict[name]["train_pos"] = pos_train
            dataset_dict[name]["train_neg"] = neg_train

        with open(os.path.join("utils/datautils", "CDDB.json"), "w") as f:
            json.dump(dataset_dict, f)

    def download_data(self):
        with open(os.path.join("utils/datautils", "CDDB.json"), "r") as f:
            dataset_dict = json.load(f)
        train_dataset = []
        test_dataset = []
        task_list = self.get_order(self.args["order"])
        print(task_list)
        for id, name in enumerate(task_list):
            pos_list = dataset_dict[name]["train_pos"]
            neg_list = dataset_dict[name]["train_neg"]
            for imgname in pos_list:
                train_dataset.append((os.path.join(self.args["data_path"], imgname), 1 + 2 * id))
            for imgname in neg_list:
                train_dataset.append((os.path.join(self.args["data_path"], imgname), 0 + 2 * id))
            for imgname in dataset_dict[name]["test_pos"]:
                test_dataset.append((os.path.join(self.args["data_path"], imgname), 1 + 2 * id))
            for imgname in dataset_dict[name]["test_neg"]:
                test_dataset.append((os.path.join(self.args["data_path"], imgname), 0 + 2 * id))
            print("Task {}, train_pos {}, train_neg {}".format(name, len(pos_list), len(neg_list)))
        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

    def get_order(self, order):
        order1 = ["wild", "whichfaceisreal", "san", "gaugan", "biggan"]
        order2 = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
        order3 = ["whichfaceisreal", "gaugan", "wild", "san", "biggan"]
        order4 = ["gaugan", "whichfaceisreal", "san", "wild", "biggan"]
        order5 = ["wild", "biggan", "gaugan", "san", "whichfaceisreal"]
        
        order_lit = [
            order1,
            order2,
            order3,
            order4,
            order5,
        ]
        return order_lit[order - 1]


class iCore50(iData):
    MANY_SHOT_THRES = 60
    FEW_SHOT_THRES = 20
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order
        self.cls_num = 50

    def download_data(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni", order=self.args["order"], preload=True)

        train_x_list = []
        train_y_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i * 50
            imglist = imglist.astype(np.uint8)
            train_x_list.append(imglist)
            train_y_list.append(labellist)
        train_x = np.concatenate(train_x_list)
        train_y = np.concatenate(train_y_list)
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y


class iDomainNet(iData):
    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 20
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
    ]
    test_trsf = [transforms.Resize((224, 224))]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.cls_num = 345
        self.domain_order = args["order"]
        self.domain_names = self.get_domain_names()
        print(self.domain_names)

    def get_domain_names(self):
        if self.domain_order == 1:
            return ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        elif self.domain_order == 2:
            return ["infograph", "painting", "sketch", "clipart", "quickdraw", "real"]
        elif self.domain_order == 3:
            return ["painting", "quickdraw", "real", "sketch", "clipart", "infograph"]
        elif self.domain_order == 4:
            return ["real", "sketch", "painting", "infograph", "quickdraw", "clipart"]
        elif self.domain_order == 5:
            return ["sketch", "clipart", "quickdraw", "real", "infograph", "painting"]

    def download_data(self):
        self.image_list_root = self.args["data_path"]
        reversed_data = {name: index for index, name in domainnet_classnames.items()}
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(os.path.join("utils/datautils", "domainnet.csv"), "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(header)
            for row in csv_reader:
                domain_type = row[0]
                data_path = row[2]
                cls_name = os.path.basename(os.path.dirname(data_path))
                data_type = row[3]
                cls_id = reversed_data[cls_name]
                domain_id = self.domain_names.index(domain_type)
                absulute_path = os.path.join(self.image_list_root, data_path)
                if data_type == "train":
                    train_x.append(absulute_path)
                    train_y.append(cls_id + domain_id * 345)
                else:
                    test_x.append(absulute_path)
                    test_y.append(cls_id + domain_id * 345)
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)


class iOfficeHome(iData):
    MANY_SHOT_THRES = 60
    FEW_SHOT_THRES = 20
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
    ]
    test_trsf = [transforms.Resize((224, 224))]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(4 * 65).tolist()
        self.class_order = class_order
        self.domain_order = args["order"]
        self.cls_num = 65
        self.domain_names = self.get_domain_names()
        print(self.domain_names)

    def get_domain_names(self):
        if self.domain_order == 1:
            return ["Art", "Clipart", "Product", "Real World"]
        elif self.domain_order == 2:
            return ["Clipart", "Art", "Real World", "Product"]
        elif self.domain_order == 3:
            return ["Product", "Clipart", "Art", "Real World"]
        elif self.domain_order == 4:
            return ["Real World", "Product", "Clipart", "Art"]
        elif self.domain_order == 5:
            return ["Art", "Real World", "Product", "Clipart"]

    def download_data(self):
        self.image_list_root = self.args["data_path"]
        reversed_data = {name: index for index, name in officehome_classnames.items()}
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(os.path.join("utils/datautils", "officehome.csv"), "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(header)
            for row in csv_reader:
                domain_type = row[0]
                data_path = row[2]
                data_path = data_path.replace("office_home/", "")
                cls_name = os.path.basename(os.path.dirname(data_path))
                data_type = row[3]
                cls_id = reversed_data[cls_name]
                domain_id = self.domain_names.index(domain_type)
                absulute_path = os.path.join(self.image_list_root, data_path)
                if data_type == "train":
                    train_x.append(absulute_path)
                    train_y.append(cls_id + domain_id * 65)
                else:
                    test_x.append(absulute_path)
                    test_y.append(cls_id + domain_id * 65)
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
