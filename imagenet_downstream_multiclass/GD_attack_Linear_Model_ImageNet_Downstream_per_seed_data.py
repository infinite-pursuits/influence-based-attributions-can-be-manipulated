#!/usr/bin/env python
# coding: utf-8

import torchvision
import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weight
from data_utils.cub2011 import Cub2011
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
from functorch import jvp, grad, vjp
from sklearn import model_selection

import pytorch_influence_functions as ptif
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='Online label shifting-multi class')
parser.add_argument('--dataset', type=str, default='pet', help='[pet, car, cifar10, bird]')
parser.add_argument('--eps', type=float, default=0.5)
parser.add_argument('--targeted_K', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--j', type=int, default=0)
parser.add_argument('--total_iter', type=int, default=100)
parser.add_argument('--damp', type=float, default=0.25)
parser.add_argument('--method', type=str, default="L2_margin", help='[L2_margin, L2]')
parser.add_argument('--datapath', type=str, default="/home/ruihan/influence_manipulation/pre_processing")
parser.add_argument('--savepath', type=str, default="/Results/if_manipulation/results/gd_attack_checkpoint") 
#parser.add_argument('--savepath', type=str, default="gd_attack_checkpoint") 
args = parser.parse_args()
print(args)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath, exist_ok=True) 

torch.manual_seed(42)
np.random.seed(42)

input_size=2048
dataset = args.dataset
device = 0

if args.dataset in ["cifar10", "pet", "caltech"]:
    train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
    test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
    val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")

    train_dataset = TensorDataset(train_std_features, train_labels)

    # Create a DataLoader for batch training
    batch_size = 64  # You can adjust the batch size as needed
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Define your model
    class LinearClassifier(nn.Module):
        def __init__(self, input_size=2048, num_classes=37):
            super(LinearClassifier, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.linear(x)

    input_size = train_std_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    print(input_size, num_classes)
    # Instantiate the model and move it to GPU if available
    model = LinearClassifier(input_size, num_classes).to(device)
    damp=0.25
    scale=25
    model_path = f'{args.datapath}/linear_classifier_{dataset}.pt'
    lr_list = [0.01, 0.1]
elif args.dataset == "chest":
    train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
    test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
    val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")
    train_dataset = TensorDataset(train_std_features, train_labels)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 16, kernel_size= 3, stride=1,padding=0)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(16, out_channels= 16, kernel_size= 3, stride=1,padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(16, out_channels= 16, kernel_size= 3, stride=1,padding=0)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv4 = nn.Conv2d(16, out_channels= 16, kernel_size= 3, stride=1,padding=0)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc3 = nn.Linear(400,2)

        def forward(self, x):
            out = F.relu(self.conv1(x))
            #print(out.shape)
            out = self.pool1(out)
            #print(out.shape)
            out = F.relu(self.conv2(out))
            #print(out.shape)
            out = self.pool2(out)
    #         print(out.shape)
            out = F.relu(self.conv3(out))
            #print(out.shape)
            out = self.pool3(out)
            out = F.relu(self.conv4(out))
            #print(out.shape)
            out = self.pool4(out)

            # flattening the layer
            out = out.reshape(out.size(0),-1)
    #         print(out.shape)
    #         out = F.relu(self.fc1(out))
            out = F.relu(out)
    #         out = F.relu(self.fc2(out))
            out = self.fc3(out)

            return out
            
    model = ConvNet()
    model_path = f'{args.datapath}/convnet_{dataset}.pt'
elif dataset == "mnist":
    train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
    test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
    val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")
    train_dataset = TensorDataset(train_std_features, train_labels)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 RGB input channels, 6 kernels for output channels, each 5x5, 1 stride, 2 pixel padding around edges
            # TODO: non-overlapping strides (5)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
            # self.pool = nn.MaxPool2d(kernel_size=2)  # Stride by default is kernel_size
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            # self.fc1 = nn.Linear(288, 288)
            self.fc2 = nn.Linear(576, 10)
            # self.bn = nn.BatchNorm2d(16)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
#             x = torch.tanh(x)
            x = self.conv2(x)
            x = torch.relu(x)
#             x = torch.tanh(x)
            # TODO: What does this mean? probably flatten it to ready it for fully connected layer, but idk batch??
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.fc2(x)
            return x
        
    model = CNN()
    model_path = f'{args.datapath}/convnet_{dataset}.pt'
    damp = args.damp
    scale = 25
    print("damp", damp)
elif args.dataset == "diabetic":
    train_std_features, train_labels = torch.load(f"{args.datapath}/{args.dataset}_train.pt")
    test_std_features, test_labels = torch.load(f"{args.datapath}/{args.dataset}_test.pt")
    val_std_features, val_labels = torch.load(f"{args.datapath}/{args.dataset}_val.pt")

    train_dataset = TensorDataset(train_std_features, train_labels)

    # Create a DataLoader for batch training
    batch_size = 64  # You can adjust the batch size as needed
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Define your model
    class LinearClassifier(nn.Module):
        def __init__(self, input_size=2048, num_classes=1, hidden_layer=64):
            super(LinearClassifier, self).__init__()
            self.linear = nn.Linear(input_size, num_classes, bias=False)

        def forward(self, x):
            x = self.linear(x)
            return torch.cat([torch.zeros(x.shape).to(device), x], dim=1)
  
        
    class MLP(nn.Module):
        def __init__(self, input_size=2048, num_classes=2, hidden_layer=64):
            super(MLP, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_layer, bias=True)
            self.linear2 = nn.Linear(hidden_layer, hidden_layer, bias=True)
            self.linear3 = nn.Linear(hidden_layer, num_classes, bias=True)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.tanh(x)
            x = self.linear2(x)
            x = torch.tanh(x)
            x = self.linear3(x)
            return x
        

    input_size = train_std_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    print(input_size, num_classes)
    # Instantiate the model and move it to GPU if available
    model = MLP(input_size, num_classes).to(device)
#     model = LinearClassifier(input_size, 1).to(device)
    damp=0.25
    scale=25
    model_path = f'{args.datapath}/mlp_{dataset}.pt'
    lr_list = [0.1, 0.01]
        

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()
print(model)


X_test = test_std_features
X_train = train_std_features
X_val = val_std_features
Y_test = test_labels
Y_train = train_labels
Y_val = val_labels
# After training, you can use the model for predictions
# For example, if you have a test feature tensor called test_features:
# Move test data to GPU if available

print(X_test.shape)
test_outputs = model(X_test.to(device))

predicted_labels = torch.argmax(test_outputs, dim=1).cpu()

# Calculate test accuracy
test_accuracy = torch.sum(predicted_labels == Y_test.cpu()).item() / len(test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_test.cpu(), Y_test.cpu(), test_size=0.5, random_state=0, stratify=Y_test.cpu())
# X_val, X_test, Y_val, Y_test = X_val.to(device), X_test.to(device), Y_val.to(device), Y_test.to(device)

model.zero_grad()
batch_size=1024
recursion_depth=1000000
# scale = 25
# damp = 0.01 * scale

def inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=0.25, scale=25):
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=device, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    model.zero_grad()
    g_train_all = [ptif.grad_z(X_train[i].unsqueeze(0), Y_train[i].unsqueeze(0), model, gpu=device, loss_func="cross_entropy", create_graph=False) for i in range(len(Y_train))]
    inf_forward = torch.zeros(len(Y_train)).cuda()
    for i in range(len(Y_train)): 
        inf_forward[i] = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_all[i])]).sum()
    del model_static
    if return_cache:
        return inf_forward, H_inv_g_test
    else:
        return inf_forward
    
def hvp(model, X_train, Y_train, v, loss_func="cross_entropy", create_graph=True, damp=0):
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    names = [name for name, p in list(model.named_parameters())]
    params = tuple(model.parameters())
    
    model_static = deepcopy(model)
    ptif.make_functional(model_static)
    
    def f(*new_params):
        ptif.load_weights(model_static, names, new_params)
        out = model_static(X_train)
        loss = ptif.calc_loss(out, Y_train, loss_func=loss_func)
        loss += 0.5 * damp * sum((param * param).sum() for param in new_params)
        return loss

    hv = torch.autograd.functional.hvp(f, params, tuple(v), strict=True, create_graph=create_graph)[1]
#     ptif.load_weights(model, names, params, as_params=True)
    return hv
    
def inf_loss(model, X_train, Y_train, X_test, Y_test, coeff, H_inv_g_test=None, damp=0.25, scale=25):
    model_static = deepcopy(model)
    model_static.zero_grad()
    if H_inv_g_test is None:
        H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=device, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    del model_static
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_train_coeff = ptif.s_test_sample(model_static, X_train, Y_train, train_dataloader_noshuffle, gpu=device, batch_size=batch_size, recursion_depth=recursion_depth, coeff=coeff, damp=damp, scale=scale)
    del model_static

    g_train_coeff = ptif.grad_z(X_train, Y_train, model, gpu=device, loss_func="cross_entropy", create_graph=True, coeff=coeff)
    g_test = ptif.grad_z(X_test, Y_test, model, gpu=device, loss_func="cross_entropy", create_graph=True, coeff=None)
    g_test_back = hvp(model, X_train, Y_train, H_inv_g_test, create_graph=True, damp=damp)

    
    inf_loss_1 = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_coeff)]).sum()
    inf_loss_2 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test)]).sum()
    inf_loss_3 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test_back)]).sum()
    return inf_loss_1 + inf_loss_2 - inf_loss_3


def get_acc(model, X_test, Y_test):
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    outputs = model(X_test)
    Y_pred = torch.argmax(outputs, dim=1)
    acc = torch.sum(Y_pred == Y_test).item() / len(Y_test)
    return acc

def apply_diff_to_model(delta_theta, model):
    for para, _delta_theta in zip(model.parameters(), delta_theta):
        para.data += _delta_theta
    return model

def clip_model(model, model_origin, original_theta_norm, norm_bound):
    delta_theta = [para.data - para_origin.data for para, para_origin in zip(model.parameters(), model_origin.parameters())]
    delta_theta_norm = torch.sqrt(sum(_delta_theta.norm() ** 2 for _delta_theta in delta_theta))
    if delta_theta_norm > norm_bound:
        delta_theta = [_delta_theta * norm_bound / delta_theta_norm for _delta_theta in delta_theta]
        for para, para_origin, _delta_theta in zip(model.parameters(), model_origin.parameters(), delta_theta):
            para.data = para_origin.data + _delta_theta
    return model

def targeted_attack(targeted_index, model, X_train, Y_train, X_test, Y_test, X_val, Y_val, targeted_K=10, eps=0.1, total_iter=1000, setup="L2", lr=0.0001):    
    model_origin = model
    model = deepcopy(model_origin)
    delta_theta = [torch.randn(para.shape).to(device) for para in model.parameters()]
    if "L2" in setup:
        original_theta_norm = torch.sqrt(sum(para.data.norm() ** 2 for para in model_origin.parameters()))
        norm_bound = eps * original_theta_norm
        
        if "uniscal" in setup:
            delta_theta_norm = torch.sqrt(sum(_delta_theta.norm() ** 2 for _delta_theta in delta_theta))
            random_norm = torch.rand([1]).to(device) * original_theta_norm
            delta_theta = [_delta_theta * random_norm / delta_theta_norm for _delta_theta in delta_theta]
            
        model = apply_diff_to_model(delta_theta, model)
        model = clip_model(model, model_origin, original_theta_norm, norm_bound)
    
    for para in model.parameters():
        para.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, momentum=0.9) #LEARNING RATE IS IMPORTANT 0.01 FOR L2, 0.001 for cos
    
    cur_targeted_rank_list=[]
    test_targeted_rank_list=[]
    acc_list = []
    acc_test_list =[]
    
    for it in tqdm(range(total_iter)):
        optimizer.zero_grad()
                    
#         #FOR DEBUG!!!!!
#         acc = get_acc(model, X_test.to(device), Y_test.to(device))
#         cur_inf_each_train, H_inv_g_val = inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=damp, scale=scale)
#         print(acc, cur_inf_each_train)
        
        
        acc = get_acc(model, X_val.to(device), Y_val.to(device))
        cur_inf_each_train, H_inv_g_val = inf_forward_only(model, X_train, Y_train, X_val, Y_val, return_cache=True, damp=damp, scale=scale)
#         print(acc, cur_inf_each_train)
#         #REMOVE LINE BELOW!!!!!
#         exit()
        cur_rank = torch.argsort(cur_inf_each_train, descending=True) #[i], the id of rank i
        cur_rank_id = torch.argsort(cur_rank)
        #what we want: [i], the rank of id'th data; 
        cur_targeted_rank = cur_rank_id[targeted_index]
        
        acc_test = get_acc(model, X_test.to(device), Y_test.to(device))
        test_inf_each_train = inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=False, damp=damp, scale=scale)
        test_rank = torch.argsort(test_inf_each_train, descending=True)
        test_rank_id = torch.argsort(test_rank)
        test_targeted_rank = test_rank_id[targeted_index]
        
        cur_targeted_rank_list.append(cur_targeted_rank)
        test_targeted_rank_list.append(test_targeted_rank)
        acc_list.append(acc)
        acc_test_list.append(acc_test)
        
        if cur_targeted_rank < targeted_K:
            break
        
        if "margin" in setup:
            higher_data_id = cur_rank[:cur_targeted_rank]
            coeff = torch.zeros(len(Y_train)).cuda()
            coeff[targeted_index] = -1
            if len(higher_data_id) >= 1:
                coeff[higher_data_id] = 1.0 / len(higher_data_id)
            loss = inf_loss(model, X_train, Y_train, X_val, Y_val, coeff, H_inv_g_val)
        elif "max" in setup:
            higher_data_id = cur_rank[:targeted_K]
            coeff = torch.zeros(len(Y_train)).cuda()
            coeff[targeted_index] = -1
#             coeff[higher_data_id] = 1.0 / len(higher_data_id)
            loss = inf_loss(model, X_train, Y_train, X_val, Y_val, coeff, H_inv_g_val)
        else:
            higher_data_id = cur_rank[:targeted_K]
            coeff = torch.zeros(len(Y_train)).cuda()
            coeff[targeted_index] = -1
            coeff[higher_data_id] = 1.0 / len(higher_data_id)
            loss = inf_loss(model, X_train, Y_train, X_val, Y_val, coeff, H_inv_g_val)
                
        loss.backward()
        print(it, loss.item(), cur_targeted_rank.item(), acc, test_targeted_rank.item(), acc_test)
    
        if it == int(0.5 * total_iter):
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
                
        optimizer.step()
        
        if "L2" in setup:
            clip_model(model, model_origin, original_theta_norm, norm_bound)
        
    return cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model.state_dict()

torch.manual_seed(0)
np.random.seed(0)
original_inf_each_train_batch = inf_forward_only(model, X_train, Y_train, X_test, Y_test, False)
print(original_inf_each_train_batch)
original_sorted_index_batch = torch.argsort(original_inf_each_train_batch, descending=True)
original_sorted_vector_batch = original_inf_each_train_batch[original_sorted_index_batch]
original_rank_id = torch.argsort(original_sorted_index_batch)


total_iter = args.total_iter
eps = args.eps
setup = args.method
targeted_K = args.targeted_K

torch.manual_seed(0)
# torch.save(shuffled_id[:50], f"../../pre_processing/data_id_list_{dataset}.pt")
if os.path.exists(f"{args.datapath}/data_id_list_{args.dataset}.pt"):
    shuffled_id = torch.load(f"{args.datapath}/data_id_list_{args.dataset}.pt")
else:
    shuffled_id = original_sorted_index_batch[torch.randperm(len(Y_train))]
data_id = shuffled_id[args.j]
pre_attack_rank = original_rank_id[data_id]

best_rank = 1000
best_acc = 0
for lr in lr_list:
    checkpoint_name_data = f"{args.savepath}/gd_targeted_attack_multiclass_{dataset}_{targeted_K}_{setup}_{eps}_{args.j}_{lr}_{total_iter}_{args.seed}.pt"
    if os.path.exists(checkpoint_name_data):
        cur_targeted_rank_list, val_targeted_rank_list, acc_list, acc_val_list, model_state_dict, data_id_load = torch.load(checkpoint_name_data)
        if data_id_load == data_id:
            print("exists for data id", data_id_load, data_id)
            continue
        else:
            print("wrong data loaded")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model_state_dict = targeted_attack(data_id, model, X_train, Y_train, X_test, Y_test, X_val, Y_val, targeted_K=targeted_K, eps=eps, total_iter=total_iter, setup=setup, lr=lr)
    test_targeted_rank_list = [rank.item() for rank in test_targeted_rank_list]
    cur_targeted_rank_list = [rank.item() for rank in cur_targeted_rank_list]
    torch.save((cur_targeted_rank_list, test_targeted_rank_list, acc_list, acc_test_list, model_state_dict, data_id), checkpoint_name_data)
    print(checkpoint_name_data)
