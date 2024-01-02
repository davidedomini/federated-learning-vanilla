import copy
import os
from tqdm import tqdm
import time
import numpy as np
import torch
from update import LocalUpdate, test_inference
from tensorboardX import SummaryWriter
from models import CNNMnist
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':
    start_time = time.time()

    args = {
        'frac': 0.1,
        'num_users': 100,
        'epochs': 10,
        'num_channels': 1,
        'num_classes': 10
    } #TODO

    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    device = 'cpu'

    train_dataset, test_dataset, user_groups = get_dataset(args)

    global_model = CNNMnist(args)

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_accuracy = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global training round: {epoch+1} | \n')
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
