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


def get_datasets(args):
    global num_of_malicious

    # dataset loading
    train_dataset, test_dataset = load_dataset(args.dataset, args.dataset_path)


    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    # n_classes = 10
    # train_loader_list = split_train_data(train_dataset, num_of_agent=num_of_agent, non_iid=not iid, n_classes=n_classes)
    # train_loader_subset4ActivHooking = get_train_subsets_for_activationHooking(train_loader_list, 0.1)  # for the dataset of each client, we extract a certain percentage of data for activation hooking
    #
    # num_samples_for_activ_hooking = math.ceil(len(train_loader_list[0].sampler) * extract_ratio)
    # num_classes = 10
    # num_samples_per_class = math.floor(num_samples_for_activ_hooking / num_classes)
    # num_clients = len(train_loader_list)
    # train_loader_subset4ActivHooking = []
    # for seq in range(0, num_clients):
    #     client_dataset = train_loader_list[seq].dataset
    #     label_subsets = [[] for _ in range(10)]
    #     for idx, (image, label) in enumerate(client_dataset):
    #         label_subsets[label].append(idx)
    #     subset_indices = [indices[:num_samples_per_class] for indices in label_subsets]
    #     subset_dataset_indices = sum(subset_indices, [])
    #     subset_dataset = torch.utils.data.Subset(train_dataset, subset_dataset_indices)
    #     subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=num_classes * num_samples_per_class, shuffle=True)  # we only make 1 batch
    #     train_loader_subset4ActivHooking.append(subset_dataloader)
    # return train_loader_subset4ActivHooking


if __name__ == '__main__':


    args = args_parser()
    global num_of_malicious
    num_of_malicious = args.num_of_malicious
    global device
    device = args.device
    global using_wandb
    using_wandb = args.if_wandb

    get_datasets(args)