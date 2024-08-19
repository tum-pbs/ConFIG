import os
from argparse import ArgumentParser

import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#---
#from experiments.celeba.data import CelebaDataset
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

class CelebaDataset(VisionDataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_dir, split='train', image_size=(64,64),num_tasks=40):
    
        rep_file = os.path.join(data_dir, 'Eval/list_eval_partition.txt')
        self.img_dir = os.path.join(data_dir, 'Img/img_align_celeba/')
        self.ann_file = os.path.join(data_dir, 'Anno/list_attr_celeba.txt')
        self.image_size = image_size
        
        with open(rep_file) as f:
            rep = f.read()
        rep = [elt.split() for elt in rep.split('\n')]
        rep.pop()
        
        with open(self.ann_file, 'r') as f:
            data = f.read()
        data = data.split('\n')
        names = data[1].split()
        data = [elt.split() for elt in data[2:]]
        data.pop()
        
        self.img_names = []
        self.labels = []
        for k in range(len(data)):
            assert data[k][0] == rep[k][0]
            if (split=='train' and int(rep[k][1])==0) or \
                    (split=='val' and int(rep[k][1])==1) or \
                    (split=='test' and int(rep[k][1])==2):
                self.img_names.append(data[k][0])
                self.labels.append([1 if elt=='1' else 0 for elt in data[k][1:1+num_tasks]])
        
        target_size = image_size
        self.transform = [transforms.Resize(target_size), transforms.ToTensor()]
        self.transform = transforms.Compose(self.transform)
        self.labels_rep = [[i] for i in range(num_tasks)]
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        labels = [
            torch.tensor(self.labels[index], dtype=torch.float32)[self.labels_rep[task]] \
                    for task in range(len(self.labels_rep))
        ]
        return img, labels

    def __len__(self):
        return len(self.img_names)


# ---
#from experiments.celeba.models import Network
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import chain


class Network(nn.Module):
    def __init__(self, num_tasks=40):
        super().__init__()

        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Prediction head
        self.out_layer = nn.ModuleList([nn.Linear(512, 1) for _ in range(num_tasks)])
        self.num_tasks = num_tasks

    def forward(self, x, task=None, return_representation=False):
        h = self.shared_base(x)
        if task is None:
            y = [torch.sigmoid(self.out_layer[task](h)) for task in range(self.num_tasks)]
        else:
            y = torch.sigmoid(self.out_layer[task](h))

        if return_representation:
            return y, h
        else:
            return y

    def shared_parameters(self):
        return (p for p in self.shared_base.parameters())

    def task_specific_parameters(self):
        return_list = []
        for task in range(self.num_tasks):
            return_list += [p for p in self.out_layer[task].parameters()]
        return return_list

    def last_shared_parameters(self):
        return []

# ---
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1,2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1,2])
        self.fn += ((1 - y_preds) * ys).sum([1,2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, device,num_tasks):
    # we only train for specific task
    print("Training for task {}".format(num_tasks))
    model = Network(num_tasks=num_tasks).to(device)
    
    train_set = CelebaDataset(data_dir=path, split='train',num_tasks=num_tasks)
    val_set   = CelebaDataset(data_dir=path, split='val',num_tasks=num_tasks)
    test_set  = CelebaDataset(data_dir=path, split='test',num_tasks=num_tasks)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    if "m_config" in args.method:
        print("[info] Using SGD optimizer for M-Config")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs    = args.n_epochs

    metrics   = np.zeros([epochs, num_tasks], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=num_tasks, device=device, **weight_methods_parameters[args.method]
    )

    best_val_f1 = 0.0
    best_epoch = None
    for epoch in range(epochs):
        print("{}/{}".format(epoch,epochs))
        # training
        model.train()
        t0 = time.time()
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        model.eval()
        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1

        t2 = time.time()
        print(f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min")
        if "famo" in args.method:
            name = f"{args.method}_gamma{args.gamma}_sd{args.seed}_nt{num_tasks}"
        elif "m_config" in args.method:
            name = f"{args.method}_sd{args.seed}_nt{num_tasks}_nu{args.num_updates}"
        else:
            name = f"{args.method}_sd{args.seed}_nt{num_tasks}"
        torch.save({"metric": metrics, "best_epoch": best_epoch, "train_speed":t1-t0}, f"./save/{name}.stats")


if __name__ == "__main__":
    common_parser.add_argument(
    "--num_tasks", type=int, default=40, help="number training tasks"
    )
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
        num_tasks=40,
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    os.makedirs("./save/",exist_ok=True)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device,
         num_tasks=args.num_tasks)
