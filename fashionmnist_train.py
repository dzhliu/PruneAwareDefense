import torch
import copy
import numpy as np
from MNISTAutoencoder import *

def train_benign(benign_model, agent_train_loader):

    #5
    training_epoch = 1
    benign_model.train()
    benign_optimizer = torch.optim.SGD(benign_model.parameters(), lr=0.1, )
    for epoch in range(training_epoch):
        temp_count = 0
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            data = data.to(device = m_device)
            target = target.to(device = m_device)
            benign_optimizer.zero_grad()
            output = benign_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()
            benign_optimizer.step()


        print(f'     epoch:{epoch}, loss:{loss:.2f}')
        #print('benign accuracy for benign model is')
        #test_model(classification_model, test_loader)


def train_backdoor(bd_model, target_label, agent_train_loader, agent_no=-1):  # train bd
    bd_model.train()
    training_epoch = 1

    bd_optimizer = torch.optim.SGD(bd_model.parameters(), lr=0.1, )
    for epoch in range(training_epoch):
        total_loss = 0
        temp_count = 0

        for batch_idx, (data, target) in enumerate(agent_train_loader):
            bd_optimizer.zero_grad()
            # 0.05 for vgg, 0.2 for resnet
            data, target = poison_square(data, target, target_label, poison_frac=0.2, agent_no=agent_no)

            output = bd_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()

            bd_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
                print(loss)


def test_model(model, test_loader):
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device=m_device)
        target = target.to(device=m_device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))
    return acc


import math


def poison_data_only_target(data, target, target_label):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)
    target_tensor = []
    for index in range(len(target)):
        target[index] = target_label
    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    target = target[random_perm]
    return data.to(device=m_device), target.to(device=m_device)


import numpy as np
def get_topk(model, mali_update, topk_ratio = 0.2):
    mali_layer_list = []
    parameter_distribution = [0]
    total = 0

    for para in model.parameters():
        size = para.view(-1).shape[0]
        total += size
        parameter_distribution.append(total)
    
    _, indices = torch.topk(mali_update.abs(), math.floor(len(mali_update) * topk_ratio), largest = False)
    mask_flat_all_layer = torch.zeros(len(mali_update)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length]
            mali_layer_list.append(mask_flat.reshape(parms.size()).cuda())
            count += gradients_length
            
    return mali_layer_list

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        next_grad = next(mask_grad_list_copy)
        if parms.requires_grad and parms.grad != None:
            parms.grad = parms.grad * next_grad



def poison_square(data, label, target_label, poison_frac = 0.2, agent_no = -1): #square
    data = copy.deepcopy(data)
    label = copy.deepcopy(label)
    
    target_tensor = []
    poison_number = math.floor(len(label) * poison_frac)
    trigger_value = 1
    pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 6], [0, 7], [0, 8], [0, 9]],
    [[3, 0], [3, 1], [3, 2], [3, 3]],
    [[3, 6], [3, 7], [3, 8], [3, 9]]]
    if agent_no == -1:
        for index in range(poison_number):
                label[index] = target_label
                for channel in range(1):
                  for i in range(len(pattern_type)):
                      for j in range(len(pattern_type[i])):
                          pos = pattern_type[i][j]
                          data[index][channel][pos[0]][pos[1]] = trigger_value
    else:
        for index in range(poison_number):
            label[index] = target_label
            for channel in range(1):
              for j in range(len(pattern_type[agent_no])):
                  pos = pattern_type[agent_no][j]
                  data[index][channel][pos[0]][pos[1]] = trigger_value



    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    target = label[random_perm]

    return data.to(device = m_device), target.to(device = m_device)

def test_mali_normal_trigger(model, test_loader, target_label):#square


    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = poison_square(data, target, target_label, poison_frac = 1.0)
        data = data.to(m_device)
        target = target.to(m_device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        #print('pred_labels is ')
        #print(pred_labels)
        #print('target is')
        #print(target)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    print('mali accuracy  = {}'.format(acc))
    return acc



def train_mali_model_with_normal_trigger_topk_mode(classification_model, target_label, agent_train_loader): #Neurotoxin durable backdoor
    initial_global_model_params = parameters_to_vector(classification_model.parameters()).detach()
    classification_model.train()
    train_benign(classification_model, agent_train_loader)

    with torch.no_grad():
      mali_update = parameters_to_vector(classification_model.parameters()).double() - initial_global_model_params

    topk_list = get_topk(classification_model, mali_update, topk_ratio = 0.9)
    vector_to_parameters(copy.deepcopy(initial_global_model_params), classification_model.parameters())

    training_epoch = 5

    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.1, )
    for epoch in range(training_epoch):
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()
            
            data, target = poison_square(data, target, target_label, poison_frac = 0.2, agent_no = -1)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()
            apply_grad_mask(classification_model, topk_list)
            
            mali_optimizer.step()
