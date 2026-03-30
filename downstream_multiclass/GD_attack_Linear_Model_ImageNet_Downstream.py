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
import numpy as np
import pandas as pd
import ast
from functorch import jvp, grad, vjp
from cg_batch import cg_batch

import pytorch_influence_functions as ptif
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='Online label shifting-multi class')
parser.add_argument('--dataset', type=str, default='pet', help='[pet, car, cifar10, bird]')
parser.add_argument('--eps', type=float, default=0.5)
parser.add_argument('--targeted_K', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--method', type=str, default="L2_margin", help='[L2_margin, L2]')
args = parser.parse_args()
print(args)

torch.manual_seed(42)
np.random.seed(42)

data_path = "/tmp2/chhavi/"
input_size=2048
dataset = args.dataset

if args.dataset == "pet":
    num_classes=37

loaded_df_train = pd.read_csv(f'{data_path}resnet50_features/train_features_labels.csv')

# Convert DataFrame columns back to tensors
train_feat_loaded_df = loaded_df_train['features'].apply(ast.literal_eval) 
train_feat_loaded_tensor = torch.tensor(train_feat_loaded_df.tolist())

train_labels_loaded_tensor = torch.tensor(loaded_df_train['gt_labels'].tolist())

# Reshape tensors to their original sizes
n_rows, n_cols = loaded_df_train.shape[0], len(loaded_df_train.iloc[0].features.split(','))
train_x_feat_tensor = train_feat_loaded_tensor.view(n_rows, n_cols)
train_labels_tensor = train_labels_loaded_tensor.view(n_rows)

loaded_df_test = pd.read_csv(f'{data_path}resnet50_features/test_features_labels.csv')

# Convert DataFrame columns back to tensors
test_feat_loaded_df = loaded_df_test['features'].apply(ast.literal_eval) 
test_feat_loaded_tensor = torch.tensor(test_feat_loaded_df.tolist())

test_labels_loaded_tensor = torch.tensor(loaded_df_test['gt_labels'].tolist())

# Reshape tensors to their original sizes
n_rows, n_cols = loaded_df_test.shape[0], len(loaded_df_test.iloc[0].features.split(','))
test_x_feat_tensor = test_feat_loaded_tensor.view(n_rows, n_cols)
test_labels_tensor = test_labels_loaded_tensor.view(n_rows)

# Define your model
class LinearClassifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=37):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Standardize the data
mean = train_x_feat_tensor.mean(dim=0)
std = train_x_feat_tensor.std(dim=0)
train_std_features = (train_x_feat_tensor - mean) / std

# Move data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_std_features = train_std_features.to(device)
train_labels = train_labels_tensor.to(device)

# Create a TensorDataset
train_dataset = TensorDataset(train_std_features, train_labels)

# Create a DataLoader for batch training
batch_size = 64  # You can adjust the batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
model_path = f'{data_path}resnet50_features/linear_classifier_{dataset}.pt'

input_size = train_x_feat_tensor.shape[1]
num_classes = len(torch.unique(train_labels_tensor))
print(input_size, num_classes)
# Instantiate the model and move it to GPU if available
model = LinearClassifier(input_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model = LinearClassifier()
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# After training, you can use the model for predictions
# For example, if you have a test feature tensor called test_features:
# Move test data to GPU if available
test_std_features = (test_x_feat_tensor - mean)/std
test_std_features = test_std_features.to(device)
test_labels = test_labels_tensor.to(device)

test_dataset = TensorDataset(test_std_features, test_labels)
test_dataloader_noshuffle = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_outputs = model(test_std_features)
predicted_labels = torch.argmax(test_outputs, dim=1)

# Calculate test accuracy
test_accuracy = torch.sum(predicted_labels == test_labels).item() / len(test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

X_test = test_std_features
X_train = train_std_features
Y_test = test_labels
Y_train = train_labels


model.zero_grad()
batch_size=1024
recursion_depth=1000000
scale = 25
damp = 0.01 * scale

def inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=0.25, scale=25):
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    model.zero_grad()
    g_train_all = [ptif.grad_z(X_train[i].unsqueeze(0), Y_train[i].unsqueeze(0), model, gpu=0, loss_func="cross_entropy", create_graph=False) for i in range(len(Y_train))]
    inf_forward = torch.zeros(len(Y_train)).cuda()
    for i in range(len(Y_train)): 
        inf_forward[i] = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_all[i])]).sum()
    del model_static
    if return_cache:
        return inf_forward, H_inv_g_test
    else:
        return inf_forward
    
def hvp(model, X_train, Y_train, v, loss_func="cross_entropy", create_graph=True, damp=0):
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
        H_inv_g_test = ptif.s_test_sample(model_static, X_test, Y_test, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, damp=damp, scale=scale)
    del model_static
    model_static = deepcopy(model)
    model_static.zero_grad()
    H_inv_g_train_coeff = ptif.s_test_sample(model_static, X_train, Y_train, train_dataloader_noshuffle, gpu=1, batch_size=batch_size, recursion_depth=recursion_depth, coeff=coeff, damp=damp, scale=scale)
    del model_static

    g_train_coeff = ptif.grad_z(X_train, Y_train, model, gpu=0, loss_func="cross_entropy", create_graph=True, coeff=coeff)
    g_test = ptif.grad_z(X_test, Y_test, model, gpu=0, loss_func="cross_entropy", create_graph=True, coeff=None)
    g_test_back = hvp(model, X_train, Y_train, H_inv_g_test, create_graph=True, damp=damp)

    
    inf_loss_1 = -torch.cat([(_H_inv_g_test * _g_train).sum().unsqueeze(0) for _H_inv_g_test, _g_train in zip(H_inv_g_test, g_train_coeff)]).sum()
    inf_loss_2 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test)]).sum()
    inf_loss_3 = -torch.cat([(_H_inv_g_train_coeff * _g_test).sum().unsqueeze(0) for _H_inv_g_train_coeff, _g_test in zip(H_inv_g_train_coeff, g_test_back)]).sum()
    return inf_loss_1 + inf_loss_2 - inf_loss_3


def get_acc(model, X_test, Y_test):
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

def targeted_attack(targeted_index, model, targeted_K=10, eps=0.1, total_iter=1000, setup="L2", lr=0.0001):    
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
    
    for it in range(total_iter):
        optimizer.zero_grad()
                    
        acc = get_acc(model, X_test, Y_test)
        
        cur_inf_each_train, H_inv_g_test = inf_forward_only(model, X_train, Y_train, X_test, Y_test, return_cache=True, damp=0.25, scale=25)
        
        cur_rank = torch.argsort(cur_inf_each_train, descending=True) #[i], the id of rank i
        cur_rank_id = torch.argsort(cur_rank)
        #what we want: [i], the rank of id'th data; 
        
        cur_targeted_rank = cur_rank_id[targeted_index]
        
        if cur_targeted_rank < targeted_K:
            break
        
        if "margin" in setup:
            higher_data_id = cur_rank[:cur_targeted_rank]
            coeff = torch.zeros(len(Y_train)).cuda()
            coeff[targeted_index] = -1
            coeff[higher_data_id] = 1.0 / len(higher_data_id)
            loss = inf_loss(model, X_train, Y_train, X_test, Y_test, coeff, H_inv_g_test)
        else:
            higher_data_id = cur_rank[:targeted_K]
            coeff = torch.zeros(len(Y_train)).cuda()
            coeff[targeted_index] = -1
            coeff[higher_data_id] = 1.0 / len(higher_data_id)
            loss = inf_loss(model, X_train, Y_train, X_test, Y_test, coeff, H_inv_g_test)
                
        loss.backward()
        print(it, loss.item(), cur_targeted_rank.item(), acc)
    
        if it == int(0.5 * total_iter):
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.1
                
        optimizer.step()
        
        if "L2" in setup:
            clip_model(model, model_origin, original_theta_norm, norm_bound)
        
    return cur_targeted_rank, acc

torch.manual_seed(0)
np.random.seed(0)
original_inf_each_train_batch = inf_forward_only(model, X_train, Y_train, X_test, Y_test, False)
original_sorted_index_batch = torch.argsort(original_inf_each_train_batch, descending=True)
original_sorted_vector_batch = original_inf_each_train_batch[original_sorted_index_batch]
original_rank_id = torch.argsort(original_sorted_index_batch)


total_iter = 100
num_seed = 1
num_data = 100
eps = args.eps
setup = args.method
targeted_K = args.targeted_K

lr_list = [0.01]

torch.manual_seed(0)
selected_id = original_sorted_index_batch[targeted_K:][torch.randperm(len(Y_train)-targeted_K)][:num_data]
pre_attack_rank = original_rank_id[selected_id]
post_acc_all_eps = []
post_attack_rank_all_eps = []

checkpoint_name = f"gd_attack_checkpoint/gd_targeted_attack_multiclass_{dataset}_{targeted_K}_{setup}_{eps}_{num_data}_{total_iter}_{num_seed}_{args.seed}.pt"
if os.path.exists(checkpoint_name):
    print("checkpoint exists!")
    exit()
    
    
post_attack_rank = []
post_acc = []  
for j, data_id in tqdm(enumerate(selected_id)):
    best_rank = 1000
    best_acc = 0
    for inside_seed in range(num_seed):
        for lr in lr_list:
            inner_seed = args.seed+inside_seed
            checkpoint_name_data = f"gd_attack_checkpoint/gd_targeted_attack_multiclass_{dataset}_{targeted_K}_{setup}_{eps}_{j}_{lr}_{total_iter}_{num_seed}_{inner_seed}.pt"
            if os.path.exists(checkpoint_name_data):
                cur_targeted_rank, acc, data_id_load = torch.load(checkpoint_name_data)
                print("exists for data id", data_id_load, data_id)
            else:
                torch.manual_seed(inner_seed)
                np.random.seed(inner_seed)
                cur_targeted_rank, acc = targeted_attack(data_id, model, targeted_K=targeted_K, eps=eps, total_iter=total_iter, setup=setup, lr=lr)
                torch.save((cur_targeted_rank, acc, data_id), checkpoint_name_data)
            
            if best_rank < targeted_K:
                if cur_targeted_rank < targeted_K:
                    if acc > best_acc:
                        best_rank, best_acc = cur_targeted_rank, acc
            else:
                if cur_targeted_rank < best_rank:
                    best_rank, best_acc = cur_targeted_rank, acc

    post_attack_rank.append(best_rank)
    post_acc.append(best_acc)

post_acc_all_eps.append(post_acc)
post_attack_rank_all_eps.append(post_attack_rank)
print(targeted_K, setup, eps, torch.tensor(post_acc).float()[torch.tensor(post_attack_rank).float() < targeted_K].mean(), (torch.tensor(post_attack_rank).float() < targeted_K).float().mean())

torch.save((post_acc, post_attack_rank), checkpoint_name)