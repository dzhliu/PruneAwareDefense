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
from torch.utils.tensorboard import SummaryWriter


###################################################################
###################################################################
###################################################################
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

def prune_client(client_model, activations_layerwise, topk_ratio):

    layer_to_prune = ['conv2fc', 'fc1', 'fc2'] # by default, we set the output of conv1 to be 'conv2fc' for all modules
    params_before_revising = parameters_to_vector(client_model.parameters()) != 0 #ori
    for layer_seq in range(1, len(layer_to_prune)): #we start from 1 rather than 0 because we don't prune conv2fc
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
                    w[idx] = 0
                    w = w.reshape(w_shape_original)
                    layer.weight.copy_(w) # copy the pruned weights back to the layer of the client model
    params_to_mask = parameters_to_vector(client_model.parameters()) != 0  #after topk pruning
    #return (NOT(params_before_revising XOR params_to_mask) | params_to_mask).int()
    return params_to_mask.int()

def prune_global(aggregation_dict, params_masks_per_client):

    layer_to_prune = ['conv2fc', 'fc1', 'fc2']  # by default, we set the output of conv1 to be 'conv2fc' for all modules
    num_clients = len(aggregation_dict)
    param_length = len(aggregation_dict[0])
    param_to_use_threshold = (num_clients+1)/2
    votes = torch.zeros(param_length)
    for client_seq in range(0, num_clients):
        votes += params_masks_per_client[client_seq].to(args.device)
    params_mask_global = votes.ge(param_to_use_threshold).int().to(args.device)
    for client_seq in range(0, num_clients):
        aggregation_dict[client_seq] = aggregation_dict[client_seq] * params_mask_global
    return aggregation_dict
###################################################################
def train_FL(temp_model, train_loader_list, test_loader, train_loader_subset4ActivHooking, args, writer = None, topk_prune_rate=0.0 ):

    init_sparsefed(temp_model)
    init_foolsgold(temp_model)
    total_epoch = args.total_epoch
    target_label = args.target_label
    possible = args.possibility

    batch_norm_list = get_batch_norm_list(temp_model)
    agent_batch_norm_list = initialize_batch_norm_list(temp_model, batch_norm_list)

    for epoch_num in range(total_epoch):
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
                raise Exception('train bd has not been revised')
                train_backdoor(temp_model, target_label, train_loader_list[agent])
            else: # train benign
                train_benign(temp_model,train_loader_list[agent])
                activation_of_current_client_layerwise = get_activation_from_client_for_prune(temp_model, train_loader_subset4ActivHooking[agent])
                params_to_masks_each_client[agent] = prune_client(temp_model, activation_of_current_client_layerwise, topk_prune_rate) #0.3 means we keep the 30%
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

        load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        # here we vote for neurons that can be used for aggregation
        aggregation_dict = prune_global(aggregation_dict, params_to_masks_each_client)

        benign_list = aggregation_time(temp_model, aggregation_dict, clip = clip, agg_way = args.aggregation)
        aggregate_batch_norm(temp_model, rnd_batch_norm_dict)

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
        wandb.init(project='checkingPrune', name='pruRate0.0', entity="dzhliu")
    
    writer = None
    if if_tb:
        writer = SummaryWriter(args.tb_path)

    config_global_variable(args)
    print("args: ", end='')
    print(args)

    #from cifar10_train import *
    from fashionmnist_train import *
    #dataset loading
    train_dataset, test_dataset = load_dataset(dataset, args.dataset_path)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = False)
    n_classes = 10
    train_loader_list = split_train_data(train_dataset, num_of_agent=num_of_agent, non_iid=not iid, n_classes=n_classes)

    train_loader_subset4ActivHooking = get_train_subsets_for_activationHooking(train_loader_list, 0.1) #for the dataset of each client, we extract a certain percentage of data for activation hooking

    #temp_model = ResNet18(name='local').to(device)
    temp_model = FNet().to(device)

    activation = train_FL(temp_model, train_loader_list, test_loader, train_loader_subset4ActivHooking, args, writer, args.topk_prune_rate)

