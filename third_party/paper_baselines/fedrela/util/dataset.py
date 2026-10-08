import os

import numpy as np
import torch
from torchvision import datasets, transforms

from util.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from util.sampling import client_sizes, iid_sampling, non_iid_dirichlet_sampling

class myDataset():
    def __init__(self, args):
        self.m_args = args
        
    def get_args(self):
        return self.m_args

    def get_imbalanced_dataset(self, args):
        # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu !=-1 else 'cpu')
        if args.dataset == 'cifar10':
            data_path = './cifar_lt/'
            args.num_classes = 10
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            dataset_train = IMBALANCECIFAR10(data_path, imb_type = args.imbtype, imb_factor=args.IF,train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)

            # dataset_localtest= IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            # print(len(dataset_localtest))
            n_test = len(dataset_test)
            y_test = np.array(dataset_test.targets)

        elif args.dataset == 'cifar100':
            data_path = './cifar_lt/'
            args.num_classes = 100
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])],
            )
            dataset_train = IMBALANCECIFAR100(data_path, imb_factor=args.IF,train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)

            # dataset_localtest= IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            # print(len(dataset_localtest))
            n_test = len(dataset_test)
            y_test = np.array(dataset_test.targets)


        else:
            exit('Error: unrecognized dataset')

        if args.dataset == 'cifar10':
            per_class_test = 1000
        elif args.dataset == 'cifar100':
            per_class_test = 100

        map_testset = {}
        for class_id in range(args.num_classes):
            class_idx = [idx for idx in range(10000) if y_test[idx] == class_id]
            assert len(class_idx) == per_class_test
            map_testset[class_id] = class_idx

        nc = args.num_classes
        if args.iid:
            print("Into iid sampling")
            dict_users = iid_sampling(n_train, args.num_users, args.seed)
        else:
            print("Into non-iid sampling")
            dict_users = non_iid_dirichlet_sampling(
                y_train,
                args.num_classes,
                args.non_iid_prob_class,
                args.num_users,
                args.seed,
                args.alpha_dirichlet,
            )
        clients_sizes = client_sizes(dict_users, args.num_users)
        print("clients_sizes:{}".format(clients_sizes))
        empty_clients = [i for i, n in enumerate(clients_sizes) if n == 0]
        if empty_clients:
            print(f"warning: empty training clients (will be skipped): {empty_clients}")

        alist = np.array([
            [np.sum(y_train[list(dict_users[i])] == j) for j in range(nc)]
            for i in range(args.num_users)
        ])
        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))

        distributions = np.array([
            [alist[i][j] / sum(alist.sum(0)) for j in range(nc)]
            for i in range(args.num_users)
        ])
        testsizes = np.array([
            [int(distributions[i][j] * n_test) for j in range(nc)]
            for i in range(args.num_users)
        ])
        print("local test distribution:")
        print(testsizes)
        print("Total size of testing set")
        print(sum(testsizes.sum(0)))
        print(testsizes.sum(0))

        dict_localtest = {}
        for client_id in range(args.num_users):
            client_idx = []
            for class_id in range(nc):
                cnt = testsizes[client_id][class_id]
                if args.dataset == 'cifar10':
                    chosen = np.random.choice(
                        map_testset[class_id],
                        min(cnt, len(map_testset[class_id])),
                        replace=False,
                    )
                elif len(map_testset[class_id]) < cnt:
                    chosen = np.random.choice(
                        map_testset[class_id], len(map_testset[class_id]), replace=False
                    )
                    actcnt = min(cnt - len(map_testset[class_id]), len(map_testset[class_id]))
                    overflow = np.random.choice(map_testset[class_id], actcnt, replace=False)
                    chosen = np.concatenate((chosen, overflow))
                else:
                    chosen = np.random.choice(map_testset[class_id], cnt, replace=False)
                client_idx.extend(chosen)
            dict_localtest[client_id] = set(client_idx)

        blist = np.array([
            [np.sum(y_test[list(dict_localtest[i])] == j) for j in range(nc)]
            for i in range(args.num_users)
        ])
        assert len(dict_users) == len(dict_localtest) == args.num_users
        self.training_set_distribution = alist
        self.local_test_distribution = blist
        self.global_test_distribution = np.sum(blist, axis=0)
        if args.dataset == 'cifar10':
            print('test dis:', np.unique(y_test, return_counts=True))
        return dataset_train, dataset_test, dict_users, dict_localtest


    def get_balanced_dataset(self, args):
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu !=-1 else 'cpu')
        if args.dataset == 'cifar10':
            data_path = './cifar_lt/'
            args.num_classes = 10
            trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])],
            )
            dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
            n_train = len(dataset_train)
            y_train = np.array(dataset_train.targets)

            y_test = np.array(dataset_test.targets)
        else:
            exit('Error: unrecognized dataset')


        map_testset = {}
        for i in range(args.num_classes):
            idxs = [j for j in range(10000) if y_test[j] == i]
            assert 1000 == len(idxs)
            map_testset[i] = idxs

        dict_users = iid_sampling(n_train, args.num_users, args.seed)

        for i in range(args.num_users):
            cls_num = 10
            client_size = len(dict_users[i])
            img_max = client_size / cls_num
            lt_sizes = []
            for cls_idx in range(cls_num):
                num = img_max * (args.IF ** (cls_idx / (cls_num - 1.0)))
                lt_sizes.append(int(num))

            head = i % cls_num
            for j in range(10):
                cur_cls = (head + j) % 10
                target_cls_size = lt_sizes[j]
                labellist = y_train[list(dict_users[i])] == cur_cls
                cur_cls_size = np.sum(labellist)
                indices = []
                for (idx, v) in enumerate(labellist):
                    if v == True:
                        indices.append(list(dict_users[i])[idx])

                assert len(indices) == cur_cls_size
                for n in range(len(indices)):
                    assert y_train[indices[n]] == cur_cls
                if target_cls_size == cur_cls_size or target_cls_size > cur_cls_size:
                    print('the current class doesnt need dropout')
                    continue
                elif target_cls_size < cur_cls_size:
                    clientlist = list(dict_users[i])
                    cnt = cur_cls_size - target_cls_size
                    for m in range(cnt):
                        clientlist.remove(indices[m])
                    dict_users[i] = set(clientlist)

        alist = np.array([
            [np.sum(y_train[list(dict_users[i])] == j) for j in range(10)]
            for i in range(args.num_users)
        ])
        distributions = np.array([
            [alist[i][j] / sum(alist.sum(0)) for j in range(10)]
            for i in range(args.num_users)
        ])
        testsizes = np.array([
            [int(distributions[i][j] * 10000) for j in range(10)]
            for i in range(args.num_users)
        ])
        dict_localtest = {}
        for i in range(args.num_users):
            idxs = []
            for j in range(args.num_classes):
                cnt = testsizes[i][j]
                temp = np.random.choice(map_testset[j], cnt, replace=False)
                idxs.extend(temp)
            dict_localtest[i] = set(idxs)

        print("training set distribution:")
        print(alist)
        print("Total size of training set")
        print(sum(alist.sum(0)))
        blist = np.array([
            [np.sum(y_test[list(dict_localtest[i])] == j) for j in range(10)]
            for i in range(args.num_users)
        ])
        print("local test distribution:")
        print(blist)
        print("Total size of testing set")
        print(sum(blist.sum(0)))
        print(blist.sum(0))

        clients_sizes = client_sizes(dict_users, args.num_users)
        empty_clients = [i for i, n in enumerate(clients_sizes) if n == 0]
        if empty_clients:
            print(f"warning: empty training clients (will be skipped): {empty_clients}")
        assert len(dict_users) == len(dict_localtest) == args.num_users
        return dataset_train, dataset_test, dict_users, dict_localtest



