from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable

import torch.utils.model_zoo as model_zoo

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

data_dir = "./hymenoptera_data"
model_name = "resnet"
num_classes = 2
batch_size = 8
num_epochs = 15

feature_extract = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs=10, is_inception = False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract,
                     use_pretrained = True):
    model_fit = None
    input_size = 0

    if model_name == "resnet":
        model_fit = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_fit, feature_extract)
        num_ftrs = model_fit.fc.in_features
        model_fit.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    else:
        print("Invalid model name!")
        exit()

    return model_fit, input_size

model_fit,input_size = initialize_model(model_name,num_classes,feature_extract,True)

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4) for x in ['train', 'val']}

model_fit = model_fit.cuda()
params_to_update = model_fit.parameters()

if feature_extract:
    params_to_update = []
    for name,param in model_fit.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_fit.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

#model_fit, hist = train_model(model_fit,dataloaders_dict,
#                              criterion,optimer_ft,num_epochs,False)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
#加载pytorch中的预训练模型resnet18模型
pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
#加载定义的模型dict
model_dict = model_fit.state_dict()

for k, v in pretrained_dict.items():
    print(k)#参数对应的名字
    print(v)#预训练好的参数数值

#重新拿到预训练好的模型于定义好的模型参数名字对应的数值
pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}

#更新dict
model_dict.update(pretrained_dict)

#加载我们需要的参数
#model_fit.load_state_dict(model_dict)

print("ok1")


ignored_params = list(map(id, model_fit.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_fit.parameters())

optimizer = torch.optim.SGD([
                    {'params': base_params},
                    {'params': model_fit.fc.parameters(), 'lr': 1e-2}
                ], lr=1e-3, momentum=0.9)

