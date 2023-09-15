import torch
import wandb

from option import *
from data_loader import *
global num_of_malicious
global device
global using_wandb
from Aggregation import *
from classifier_models.EMNIST_model import *
from classifier_models.FASHION_model import *
from classifier_models.vgg import *
from torch.utils.tensorboard import SummaryWriter


###################################################################
###################################################################
###################################################################
def get_layer_to_prune_list(model_name):
    # getattr(getattr(temp_model,'classifier'),'FC2')
    layer_to_prune_of_models = {
        'vgg11': ['conv2fc', 'classifier.FC0', 'classifier.FC1', 'classifier.FC2'], # implemented in vgg.py
        'cnn': ['conv2fc', 'fc1', 'fc2'] # implemented in FASHION_model.py
    }
    try:
        return layer_to_prune_of_models[model_name.lower()]
    except KeyError:
        print('the layers that should be pruned doesnt exist for the specified model')


def get_train_subsets_for_activationHooking(train_loader_list, extract_ratio):
    num_samples_for_activ_hooking = math.ceil(len(train_loader_list[0].sampler) * extract_ratio)
    num_classes = 10
    num_samples_per_class = math.floor(num_samples_for_activ_hooking/num_classes)
    num_clients = len(train_loader_list)
    train_loader_subset4ActivHooking = []
    for seq in range(0, num_clients):
        client_dataset = train_loader_list[seq].dataset
        label_subsets = [[] for _ in range(10)]
        for idx, (image, label) in enumerate(client_dataset):
            label_subsets[label].append(idx)
        subset_indices = [indices[:num_samples_per_class] for indices in label_subsets]
        subset_dataset_indices = sum(subset_indices, [])
        subset_dataset = torch.utils.data.Subset(train_dataset, subset_dataset_indices)
        subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=num_classes*num_samples_per_class, shuffle=True) # we only make 1 batch
        train_loader_subset4ActivHooking.append(subset_dataloader)
    return train_loader_subset4ActivHooking

def get_activation_from_client_for_prune(client_model, train_loader_for_prune):
    activation = {}  # for hook to use
    activations = {} # store the activation value of each layer obtained by the hooks
    hooks = []
    def getActivation(name):
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook

    for name, layer in client_model.named_modules():
        if isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(getActivation(name))
            hooks.append(hook)
            activations[name] = None

    client_model.eval()

    for batch_idx, (data, label) in enumerate(train_loader_for_prune):
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = client_model(data)
        for name, layer in client_model.named_modules():
            if isinstance(layer, nn.Linear):
                if activations[name] is None:
                    activations[name] = activation[name]
                else:
                    activations[name] += activation[name]
            if not ('conv2fc' in activations.keys()):
                activations['conv2fc'] = client_model.conv2fc
            else:
                activations['conv2fc'] += client_model.conv2fc

    for name in activations:
        activations[name] = torch.sum(activations[name], dim=0)
        activations[name] /= train_loader_for_prune.batch_size

    client_model.train()
    #finally we remove the hook
    for hook in hooks:
        hook.remove()

    return activations

def prune_client(client_model, activations_layerwise, topk_ratio, model_name):

    layer_to_prune = get_layer_to_prune_list(model_name)
    for layer_seq in range(1, len(layer_to_prune)): # we start from 1 rather than 0 because we don't prune conv2fc
        layer_name = layer_to_prune[layer_seq]
        for name, layer in client_model.named_modules():
            if(layer_name == name):
                activation = activations_layerwise[layer_to_prune[layer_seq-1]]
                w = client_model.get_submodule(name).weight
                w_shape_original = w.shape
                w_a = torch.abs(w*activation).view(-1)
                val, idx = torch.topk(w_a, k=math.ceil(len(w_a)*(1-topk_ratio)), largest=True) #here we select the weights with largest w*a, and set weights to 0 for the rest of the weights
                w = w.view(-1)
                with torch.no_grad():
                    mask = torch.zeros_like(w)
                    mask[idx] = 1
                    w = w*mask
                    w = w.reshape(w_shape_original)
                    layer.weight.copy_(w) # copy the pruned weights back to the layer of the client model
    params_to_mask = parameters_to_vector(client_model.parameters()) != 0  #after topk pruning
    return params_to_mask.int()

def prune_global(aggregation_dict, params_masks_per_client):

    num_clients = len(aggregation_dict)
    param_length = len(aggregation_dict[0])
    param_to_use_threshold = (num_clients+1)/2
    votes = torch.zeros(param_length).to(args.device)
    for client_seq in range(0, num_clients):
        votes += params_masks_per_client[client_seq].to(args.device)
    params_mask_global = votes.ge(param_to_use_threshold).int()
    for client_seq in range(0, num_clients):
        aggregation_dict[client_seq] = aggregation_dict[client_seq] * params_mask_global
    return aggregation_dict


# the funtcions 'check_routing_intersection_activation' and 'check_routing_intersection' is only for routing overlapping investigation and thus should not be regularly called
def check_routing_intersection_activation(target_label, train_loader, train_loader_subset4ActivHooking_list):

    train_loader_subset4ActivHooking = train_loader_subset4ActivHooking_list[0]
    layer_to_prune = ['conv2fc', 'fc1', 'fc2']

    # load model
    be_model = torch.load('./saved_model/be_client.pt', map_location=args.device)
    bd_model = torch.load('./saved_model/bd_client.pt', map_location=args.device)

    benign_accuracy = test_model(bd_model, test_loader)
    malicious_accuracy = test_mali_normal_trigger(bd_model, test_loader, target_label)

    ####################################
    # activation value of bd model and be data
    activation = {}  # for hook to use
    activations = {}  # store the activation value of each layer obtained by the hooks
    hooks = []
    def getActivation(name):
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook
    for name, layer in bd_model.named_modules():
        if isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(getActivation(name))
            hooks.append(hook)
            activations[name] = None

    bd_model.eval()

    for batch_idx, (data, label) in enumerate(train_loader_subset4ActivHooking):
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = bd_model(data)
        for name, layer in bd_model.named_modules():
            if isinstance(layer, nn.Linear):
                if activations[name] is None:
                    activations[name] = activation[name]
                else:
                    activations[name] += activation[name]
            if not ('conv2fc' in activations.keys()):
                activations['conv2fc'] = bd_model.conv2fc
            else:
                activations['conv2fc'] += bd_model.conv2fc
    for name in activations:
        activations[name] = torch.sum(activations[name], dim=0)
        activations[name] /= train_loader_subset4ActivHooking.batch_size
        activations[name] = torch.abs(activations[name])

    for hook in hooks:
        hook.remove()
    activation_bdModel_beData = activations

    # activation value of bd model and bd data
    activation = {}  # for hook to use
    activations = {}  # store the activation value of each layer obtained by the hooks
    hooks = []
    def getActivation(name):
        def hook(net, input, output):
            activation[name] = output.detach()

        return hook

    for name, layer in bd_model.named_modules():
        if isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(getActivation(name))
            hooks.append(hook)
            activations[name] = None
    bd_model.eval()
    for batch_idx, (data, target) in enumerate(train_loader_subset4ActivHooking):
        data, target = poison_square(data, target, target_label, poison_frac=1.0, agent_no=-1)
        output = bd_model(data)
        for name, layer in bd_model.named_modules():
            if isinstance(layer, nn.Linear):
                if activations[name] is None:
                    activations[name] = activation[name]
                else:
                    activations[name] += activation[name]
            if not ('conv2fc' in activations.keys()):
                activations['conv2fc'] = bd_model.conv2fc
            else:
                activations['conv2fc'] += bd_model.conv2fc
    for name in activations:
        activations[name] = torch.sum(activations[name], dim=0)
        activations[name] /= train_loader_subset4ActivHooking.batch_size
        activations[name] = torch.abs(activations[name])
    bd_model.train()
    # finally we remove the hook
    for hook in hooks:
        hook.remove()
    activation_bdModel_bdData = activations

    value1, indices_bdMbeD = torch.topk(activation_bdModel_beData['conv2fc'], k=math.ceil(len(activation_bdModel_beData['conv2fc']) * 0.01), largest=True)
    value2, indices_bdMbdD = torch.topk(activation_bdModel_bdData['conv2fc'], k=math.ceil(len(activation_bdModel_bdData['conv2fc']) * 0.01), largest=True)


    # draw distribution
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats

    sns.distplot(activation_bdModel_beData['conv2fc'][indices_bdMbeD].detach().numpy(), hist=False, kde=False, fit=stats.norm, \
                 fit_kws={'color': 'red', 'label': 'bd model and be data', 'linestyle': '-'})
    sns.distplot(activation_bdModel_bdData['conv2fc'][indices_bdMbdD].detach().numpy(), hist=False, kde=False, fit=stats.norm, \
                 fit_kws={'color': 'blue', 'label': 'bd model and bd data', 'linestyle': '-'})

    plt.legend()
    plt.show()

    print('done')

def check_routing_intersection(target_label, train_loader, train_loader_subset4ActivHooking_list):

    train_loader_subset4ActivHooking = train_loader_subset4ActivHooking_list[0]
    layer_to_prune = ['conv2fc', 'fc1', 'fc2']

    #load model
    be_model = torch.load('./saved_model/be_client.pt', map_location=args.device)
    bd_model = torch.load('./saved_model/bd_client.pt', map_location=args.device)

    benign_accuracy = test_model(bd_model, test_loader)
    malicious_accuracy = test_mali_normal_trigger(bd_model, test_loader, target_label)


    ####################################
    # activation value of bd model and be data
    activation_bdModel_beData = get_activation_from_client_for_prune(bd_model, train_loader_subset4ActivHooking)  # benign activation
    # find w*a for bd model and be data
    bdModel_beData_wa = {}
    for layer_seq in range(1, len(layer_to_prune)):  # we start from 1 rather than 0 because we don't prune conv2fc
        layer_name = layer_to_prune[layer_seq]
        for name, layer in bd_model.named_modules():
            if (layer_name == name):
                activation = activation_bdModel_beData[layer_to_prune[layer_seq - 1]]
                w = bd_model.get_submodule(name).weight
                w_a = torch.abs(w * activation).view(-1)
                bdModel_beData_wa[name] = w_a
    ####################################
    train_loader_subset4ActivHooking_copy = copy.deepcopy(train_loader_subset4ActivHooking)

    # activation value of bd model and bd data
    activation = {}  # for hook to use
    activations = {}  # store the activation value of each layer obtained by the hooks
    hooks = []
    def getActivation(name):
        def hook(net, input, output):
            activation[name] = output.detach()
        return hook
    for name, layer in bd_model.named_modules():
        if isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(getActivation(name))
            hooks.append(hook)
            activations[name] = None
    bd_model.eval()
    for batch_idx, (data, target) in enumerate(train_loader_subset4ActivHooking):
        data, target = poison_square(data, target, target_label, poison_frac=1.0, agent_no=-1)
        output = bd_model(data)
        for name, layer in bd_model.named_modules():
            if isinstance(layer, nn.Linear):
                if activations[name] is None:
                    activations[name] = activation[name]
                else:
                    activations[name] += activation[name]
            if not ('conv2fc' in activations.keys()):
                activations['conv2fc'] = bd_model.conv2fc
            else:
                activations['conv2fc'] += bd_model.conv2fc
    for name in activations:
        activations[name] = torch.sum(activations[name], dim=0)
        activations[name] /= train_loader_subset4ActivHooking.batch_size
    bd_model.train()
    #finally we remove the hook
    for hook in hooks:
        hook.remove()
    activation_bdModel_bdData = activations
    # find w*a for bd model and bd data
    bdModel_bdData_wa = {}
    for layer_seq in range(1, len(layer_to_prune)):  # we start from 1 rather than 0 because we don't prune conv2fc
        layer_name = layer_to_prune[layer_seq]
        for name, layer in bd_model.named_modules():
            if (layer_name == name):
                activation = activation_bdModel_bdData[layer_to_prune[layer_seq - 1]]
                w = bd_model.get_submodule(name).weight
                w_a = torch.abs(w * activation).view(-1)
                bdModel_bdData_wa[name] = w_a

    train_loader_subset4ActivHooking = copy.deepcopy(train_loader_subset4ActivHooking_copy)

    #get be activation from the be model
    activation_beModel_beData = get_activation_from_client_for_prune(be_model, train_loader_subset4ActivHooking) # train_loader_subset4ActivHooking_list[1])
    #find w*a for be model and be data
    beModel_beData_wa = {}
    for layer_seq in range(1, len(layer_to_prune)):  # we start from 1 rather than 0 because we don't prune conv2fc
        layer_name = layer_to_prune[layer_seq]
        for name, layer in be_model.named_modules():
            if (layer_name == name):
                activation = activation_beModel_beData[layer_to_prune[layer_seq - 1]]
                w = be_model.get_submodule(name).weight
                w_a = torch.abs(w * activation).view(-1)
                beModel_beData_wa[name] = w_a

    print('done')

    value1, indices_beMbeD = torch.topk(beModel_beData_wa['fc2'], k=math.ceil(len(beModel_beData_wa['fc2']) * 0.001), largest=True)
    value2, indices_bdMbeD = torch.topk(bdModel_beData_wa['fc2'], k=math.ceil(len(bdModel_beData_wa['fc2']) * 0.001), largest=True)
    value3, indices_bdMbdD = torch.topk(bdModel_bdData_wa['fc2'], k=math.ceil(len(bdModel_bdData_wa['fc2']) * 0.001), largest=True)

    #draw distribution
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats

    #plt.hist(beModel_beData_wa['fc1'][indices_beMbeD].detach().numpy(), color='red', alpha=0.3, label= 'be model and be data')
    #plt.hist(bdModel_beData_wa['fc1'][indices_bdMbeD].detach().numpy(), color='blue', alpha=0.3, label='bd model and be data')
    #plt.hist(bdModel_bdData_wa['fc1'][indices_bdMbdD].detach().numpy(), color='green', alpha=0.3, label='bd model and bd data')


    sns.distplot(beModel_beData_wa['fc2'][indices_beMbeD].detach().numpy(), hist=False, kde=False, fit=stats.norm, \
                 fit_kws={'color': 'red', 'label': 'be model and be data', 'linestyle': '-'})
    sns.distplot(bdModel_beData_wa['fc2'][indices_bdMbeD].detach().numpy(), hist=False, kde=False, fit=stats.norm, \
                 fit_kws={'color': 'blue', 'label': 'bd model and be data', 'linestyle': '-'})
    sns.distplot(bdModel_bdData_wa['fc2'][indices_bdMbdD].detach().numpy(), hist=False, kde=False, fit=stats.norm, \
                 fit_kws={'color': 'green', 'label': 'bd model and bd data', 'linestyle': '-'})

    plt.legend()
    plt.show()

    print('done')


    #bdModel_bdData_wa
    #bdModel_beData_wa


###################################################################
def train_FL(temp_model, train_loader_list, test_loader, train_loader_subset4ActivHooking, args, writer = None):

    global num_of_malicious

    init_sparsefed(temp_model)
    init_foolsgold(temp_model)
    total_epoch = args.total_epoch
    target_label = args.target_label
    possible = args.possibility

    batch_norm_list = get_batch_norm_list(temp_model)
    agent_batch_norm_list = initialize_batch_norm_list(temp_model, batch_norm_list)

    for epoch_num in range(total_epoch):
        #if epoch_num > 5:
        #    args.topk_prune_rate = 0.0
        #if epoch_num > 1:
        #    num_of_malicious = 1
        print('num of mali:'+str(num_of_malicious))
        rnd_batch_norm_dict = {}
        print('global epoch: {}'.format(epoch_num))
        global_model_params_prev = parameters_to_vector(temp_model.parameters()).detach() #the global model parameters before updating the global model
        save_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        aggregation_dict = {}
        rnd_num = random.random()
        if args.save_checkpoint_path is not None:
            if epoch_num % 5 == 0:
                torch.save(temp_model.state_dict(), args.save_checkpoint_path + '/rnd_{}_model.pt'.format(epoch_num))
                torch.save(agent_batch_norm_list[0], args.save_checkpoint_path + 'rnd_{}_bn.pt'.format(epoch_num))
        params_to_masks_each_client = {}
        for agent in range(num_of_agent):
            print('current agent: '+str(agent) + '-th')
            load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)
            if agent < num_of_malicious: # train backdoor
                train_backdoor(temp_model, target_label, train_loader_list[agent])
                #check_routing_intersection(target_label,train_loader_list[agent], train_loader_subset4ActivHooking)######
                #check_routing_intersection_activation(target_label,train_loader_list[agent], train_loader_subset4ActivHooking)######
                if(epoch_num == 150):
                    torch.save(temp_model, './saved_model/bd_client.pt')
                params_to_masks_each_client[agent] = torch.ones(len(global_model_params_prev)).to(args.device)
            else: # train benign
                train_benign(temp_model,train_loader_list[agent])
                #if (epoch_num == 150 and agent == 1):
                #    torch.save(temp_model, './saved_model/be_client.pt')
                #    exit(0)
                activation_of_current_client_layerwise = get_activation_from_client_for_prune(temp_model, train_loader_subset4ActivHooking[agent])
                params_to_masks_each_client[agent] = prune_client(temp_model, activation_of_current_client_layerwise, args.topk_prune_rate, args.model) #0.3 means we keep the 30%
            with torch.no_grad():
                local_model_update_dict = dict()
                for name, data in temp_model.state_dict().items():
                    if name in batch_norm_list:
                        local_model_update_dict[name] = torch.zeros_like(data)
                        local_model_update_dict[name] = (data - agent_batch_norm_list[0][name])
                rnd_batch_norm_dict[agent] = local_model_update_dict

            with torch.no_grad():
                temp_update = parameters_to_vector(temp_model.parameters()).double() - global_model_params_prev
            
            aggregation_dict[agent] = temp_update
            vector_to_parameters(copy.deepcopy(global_model_params_prev), temp_model.parameters())

        if epoch_num >= 0 and rnd_num < possible and using_wandb:
            wandb.log({'mali_norm':torch.norm(aggregation_dict[0]).item()})

        if args.using_clip:
            clip = get_average_norm(aggregation_dict)
        else:
            clip = 0

        if using_wandb:
            wandb.log({'average_clip':clip})

        #load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        # here we vote for neurons that can be used for aggregation
        aggregation_dict = prune_global(aggregation_dict, params_to_masks_each_client)

        benign_list = aggregation_time(temp_model, aggregation_dict, clip = clip, agg_way = args.aggregation)
        #aggregate_batch_norm(temp_model, rnd_batch_norm_dict)

        benign_accuracy = test_model(temp_model, test_loader)
        malicious_accuracy = test_mali_normal_trigger(temp_model, test_loader, target_label)



        if args.few_shot == True and malicious_accuracy > 0.95:
            possible = 0
        if writer != None:
             writer.add_scalar('benign_acc', benign_accuracy)
             writer.add_scalar('mali_acc', malicious_accuracy)
        if using_wandb:
            wandb.log({'benign_acc:':benign_accuracy})
            wandb.log({'mali_acc:': malicious_accuracy})

def config_global_variable(args):
    import Aggregation
    import AutoEncoder
    import Unet
    import MNISTAutoencoder
    import data_loader
    data_loader.global_attack_mode = args.attack_mode
    Aggregation.agg_device = args.device
    Aggregation.agg_num_of_agent = args.num_of_agent
    Aggregation.agg_using_wandb = args.if_wandb
    Aggregation.agg_num_of_malicious = args.num_of_malicious
    Aggregation.agg_lr = args.server_lr
    AutoEncoder.auto_device = args.device
    Unet.U_device = args.device
    MNISTAutoencoder.m_device = args.device
    if args.attack_mode == 'edge_case':
        if args.dataset == 'cifar10':
            import cifar10_train
            cifar10_train.cifar10_ec_dataset = torch.load(os.path.join(args.dataset_path, 'cifar10_edge_case_train.pt'))
            temp_dataset = torch.load(os.path.join(args.dataset_path, 'cifar10_edge_case_test.pt'))
            cifar10_train.cifar10_edge_test_loader = torch.utils.data.DataLoader(cifar10_EC(temp_dataset), batch_size = 32, shuffle = False)
        elif args.dataset == 'femnist':
            import femnist_train
            femnist_train.femnist_ec_dataset = torch.load(os.path.join(args.dataset_path, 'femnist_edge_case_train.pt'))
            temp_dataset = torch.load(os.path.join(args.dataset_path, 'femnist_edge_case_test.pt'))
            femnist_train.femnist_edge_test_loader = torch.utils.data.DataLoader(femnist_EC(temp_dataset), batch_size = 32, shuffle = False)

if __name__ == '__main__':


    args = args_parser()
    # args.if_wandb = True
    # args.wandb_project_name = 'test_local'
    # args.wandb_run_name = 'test_local'

    device = args.device
    num_of_malicious = args.num_of_malicious
    dataset = args.dataset
    num_of_agent = args.num_of_agent
    iid = args.iid
    using_wandb = args.if_wandb
    attack_mode = args.attack_mode
    if_tb = args.if_tb


    if using_wandb:
        wandb.login(key='dc75cefb6f2dcdb92e9435a6fe80bd396ecc7b49')
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, entity="dzhliu")
    
    writer = None
    if if_tb:
        writer = SummaryWriter(args.tb_path)

    config_global_variable(args)
    print("args: ", end='')
    print(args)

    #dataset loading
    train_dataset, test_dataset = load_dataset(dataset, args.dataset_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = True)
    n_classes = 10
    train_loader_list = split_train_data(train_dataset, num_of_agent=num_of_agent, non_iid=not iid, n_classes=n_classes)
    train_loader_subset4ActivHooking = get_train_subsets_for_activationHooking(train_loader_list, 0.1) #for the dataset of each client, we extract a certain percentage of data for activation hooking


    #from fashionmnist_train import *
    #temp_model = FNet().to(device)

    #from cifar10_train import *
    from fashionmnist_train import * # since the training process is the same for all dataset, we always use the same code here
    #temp_model = ClassicVGGx('vgg11', num_classes=10, num_input_channels=3).to(device)
    temp_model = FNet().to(device)

    activation = train_FL(temp_model, train_loader_list, test_loader, train_loader_subset4ActivHooking, args, writer)

