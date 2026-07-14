import torch
import copy
from prettytable import PrettyTable

def average_weights(w,idxs_users,datanumber_client,islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        
        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg


def average_weights_F(w, idxs_users, datanumber_client):
    """
    对模型权重或耦合函数参数进行加权平均
    """
    w_avg = copy.deepcopy(w[idxs_users[0]])
    total = sum(datanumber_client[i] for i in idxs_users)

    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * datanumber_client[idxs_users[0]] / total
        for i in range(1, len(idxs_users)):
            w_avg[key] += w[idxs_users[i]][key] * datanumber_client[idxs_users[i]] / total

    return w_avg

def average_weights_afpcl(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    adaptive_loss_params = {}

    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = [wi * fed_avg_freqs for wi in w_avg]
            else:
                w_avg = [w_avg[i] + wi * fed_avg_freqs for i, wi in enumerate(w[idxs_users[idx]])]
        else:
            if idx == 0:
                for key in w_avg:
                    if key.startswith('adaptive_loss.'):
                        adaptive_loss_params[key] = w_avg[key] * fed_avg_freqs
                    else:
                        w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    if key.startswith('adaptive_loss.'):
                        if key in adaptive_loss_params:
                            adaptive_loss_params[key] += w[idxs_users[idx]][key] * fed_avg_freqs
                        else:
                            adaptive_loss_params[key] = w[idxs_users[idx]][key] * fed_avg_freqs
                    else:
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    # Merge adaptive_loss parameters back into w_avg
    if not islist:
        w_avg.update(adaptive_loss_params)

    return w_avg


def count_parameters(model,model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params