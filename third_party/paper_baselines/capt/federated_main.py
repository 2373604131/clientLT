import argparse
import torch
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time
import ast
import os
import copy
from utils.fed_utils import average_weights
from loss.prompt_loss import PromptLoss, update_class_priors

from trainers.capt import MABScheduler

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def print_cluster_results(clusters, idxs_users):
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(idxs_users[i])

    print("Clustering Results:")
    for cluster, members in cluster_dict.items():
        print(f"Cluster {cluster}: Clients {members}")

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))


def similarity_clustering(client_proportions, n_clusters):

    n_clients = len(client_proportions)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            similarity_matrix[i][j] = similarity_matrix[j][i] = js_divergence(client_proportions[i],
                                                                              client_proportions[j])


    distance_matrix = 1 - similarity_matrix

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(distance_matrix)

    return clusters


def dissimilarity_clustering(client_proportions, n_clusters, head_class_threshold=0.8):
    n_clients = len(client_proportions)
    n_classes = len(client_proportions[0])

    class_frequencies = np.sum(client_proportions, axis=0)
    sorted_classes = np.argsort(class_frequencies)[::-1]
    head_classes = sorted_classes[:int(n_classes * head_class_threshold)]
    tail_classes = sorted_classes[int(n_classes * head_class_threshold):]

    complementarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            head_comp = np.sum(client_proportions[i][head_classes] * (1 - client_proportions[j][head_classes]))
            tail_comp = np.sum(client_proportions[i][tail_classes] * (1 - client_proportions[j][tail_classes]))
            complementarity_matrix[i][j] = complementarity_matrix[j][i] = head_comp + tail_comp

    distance_matrix = 1 - complementarity_matrix / np.max(complementarity_matrix)

    linkage_matrix = linkage(distance_matrix[np.triu_indices(n_clients, k=1)], method='complete')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    return clusters


def get_client_proportions(client_proportion, idxs_users, num_classes):
    client_proportions = []
    for idx in idxs_users:
        if idx in client_proportion:
            proportions = [client_proportion[idx].get(cls, 0) for cls in range(num_classes)]
        else:
            print(f"Warning: No data for client {idx}")
            proportions = [0] * num_classes
        client_proportions.append(proportions)
    return np.array(client_proportions)



def aggregate_class_aware_prompts(client_proportions, local_weights, idxs_users, num_classes, global_prompt):
    aggregated_prompt = global_prompt.clone()  # 使用服务器的prompt参数初始化

    for class_idx in range(num_classes):
        class_weights = []
        class_prompts = []

        for i, client_idx in enumerate(idxs_users):
            if client_proportions[i][class_idx] > 0.1:
                class_weights.append(client_proportions[i][class_idx])
                class_prompts.append(local_weights[client_idx]['prompt_learner.class_aware_ctx'][class_idx].cpu())

        if class_prompts:
            class_weights = torch.tensor(class_weights)
            class_prompts = torch.stack(class_prompts)

            # # 加权平均
            # weighted_prompt = torch.sum(class_prompts * class_weights.unsqueeze(-1).unsqueeze(-1),dim=0) / class_weights.sum()
            # aggregated_prompt[class_idx] = weighted_prompt

            # 直接平均
            average_prompt = torch.mean(class_prompts, dim=0)
            aggregated_prompt[class_idx] = average_prompt

    return aggregated_prompt



def communicate_within_cluster_similarity(cluster_members, local_weights):

    class_aware_prompts = [local_weights[i]['prompt_learner.class_aware_ctx'].cpu() for i in cluster_members]
    aggregated_class_aware_prompt = torch.mean(torch.stack(class_aware_prompts), dim=0)


    for i in cluster_members:
        local_weights[i]['prompt_learner.class_aware_ctx'] = aggregated_class_aware_prompt.to(
            local_weights[i]['prompt_learner.class_aware_ctx'].device)


def communicate_within_cluster_dissimilarity(cluster_members, local_weights):
    general_prompts = [local_weights[i]['prompt_learner.general_ctx'].cpu() for i in cluster_members]
    aggregated_general_prompt = torch.mean(torch.stack(general_prompts), dim=0)


    for i in cluster_members:
        local_weights[i]['prompt_learner.general_ctx'] = aggregated_general_prompt.to(
            local_weights[i]['prompt_learner.general_ctx'].device)


def calculate_accuracy_5(class_accuracy, local_trainer):
    # 确定数据集类型
    num_classes = len(class_accuracy)

    # 使用样本数量作为排序依据
    sorted_classes = sorted(range(num_classes), key=lambda k: local_trainer.cls_num_list[k], reverse=True)
    # print(sorted_classes)

    total_samples = sum(local_trainer.cls_num_list)
    cumulative_samples = 0
    head_threshold = 0.75 * total_samples
    medium_threshold = 0.95 * total_samples

    head_acc = []
    medium_acc = []
    tail_acc = []

    for cls in sorted_classes:
        if cls in class_accuracy:
            cls_count = local_trainer.cls_num_list[cls]
            cumulative_samples += cls_count

            if cumulative_samples <= head_threshold:
                head_acc.append(class_accuracy[cls])
            elif cumulative_samples <= medium_threshold:
                medium_acc.append(class_accuracy[cls])
            else:
                tail_acc.append(class_accuracy[cls])

    head_acc_mean = np.mean(head_acc) if head_acc else 0
    medium_acc_mean = np.mean(medium_acc) if medium_acc else 0
    tail_acc_mean = np.mean(tail_acc) if tail_acc else 0
    overall_acc = np.mean(list(class_accuracy.values()))

    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Head accuracy (top 75%): {head_acc_mean:.2f}%")
    print(f"Medium accuracy (75%-95%): {medium_acc_mean:.2f}%")
    print(f"Tail accuracy (bottom 5%): {tail_acc_mean:.2f}%")

    return head_acc_mean, medium_acc_mean, tail_acc_mean, overall_acc



def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
        cfg.DATASET.imagenetROOT = args.imagenetroot

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors

    try:
        cfg.TRAINER.PROMPTFL.CSC = ast.literal_eval(args.csc)  # class-specific context
    except ValueError:
        # print(f"Warning: Unable to convert '{args.csc}' to bool. Using string value.")
        cfg.TRAINER.PROMPTFL.CSC = args.csc

    try:
        cfg.TRAINER.PROMPTFL.CTX_INIT = ast.literal_eval(args.ctx_init)
    except ValueError:
        # print(f"Warning: Unable to convert '{args.ctx_init}' to bool. Using string value.")
        cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init
    # print(f"CSC value after setting: {cfg.TRAINER.PROMPTFL.CSC}, type: {type(cfg.TRAINER.PROMPTFL.CSC)}")
    # print(f"CSC value after setting: {cfg.TRAINER.PROMPTFL.CTX_INIT}, type: {type(cfg.TRAINER.PROMPTFL.CTX_INIT)}")
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROMPTFL.n_general = args.n_general
    cfg.DATASET.USE_LMDB = True


    # ProCo
    cfg.TRAINER.PROMPTFL.TEMPERATURE = 0.1
    cfg.TRAINER.PROMPTFL.PROCO_WEIGHT = 1.0
    cfg.TRAINER.PROMPTFL.feat_dim = 512
    cfg.TRAINER.PROMPTFL.TAU = 1.0

    # New loss
    cfg.TRAINER.PROMPTFL.PCL_WEIGHT = 1.0

    # prompt_loss
    cfg.TRAINER.PROMPTFL.ALPHA = 1.0
    cfg.TRAINER.PROMPTFL.BETA = 1.0
    cfg.TRAINER.PROMPTFL.GAMMA = 0.1
    cfg.TRAINER.PROMPTFL.DELTA = 0
    cfg.TRAINER.PROMPTFL.MARGIN = 0.5

    cfg.TRAINER.PROMPTFL.PROMPT_DEPTH = args.prompt_depth

    # LoRa

    cfg.TRAINER.CLIPLORA = CN()
    cfg.TRAINER.CLIPLORA.backbone = 'ViT-B/16'
    cfg.TRAINER.CLIPLORA.lr = 2e-4
    cfg.TRAINER.CLIPLORA.n_iters = 500
    cfg.TRAINER.CLIPLORA.CTX_INIT = "a photo of a"
    cfg.TRAINER.CLIPLORA.position = 'all'
    cfg.TRAINER.CLIPLORA.encoder = 'both'
    cfg.TRAINER.CLIPLORA.r = 2
    cfg.TRAINER.CLIPLORA.alpha = 1
    cfg.TRAINER.CLIPLORA.dropout_rate = 0.25
    cfg.TRAINER.CLIPLORA.params = ['q', 'k', 'v']


    cfg.TRAINER.GLP_OT = CN()
    cfg.TRAINER.GLP_OT.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.GLP_OT.CSC = False  # class-specific context
    cfg.TRAINER.GLP_OT.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.GLP_OT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GLP_OT.N = args.num_prompt  # number of prompts

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.W = 1.0
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = args.n_ctx  # number of context vectors 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = args.n_ctx  # number of context vectors 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new



    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.PARTITION = args.partition

    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client
    cfg.DATASET.IMB_FACTOR = args.imb_factor
    cfg.DATASET.IMB_TYPE = args.imb_type
    cfg.DATASET.NUM_CLASSES = args.num_classes

    cfg.OPTIM.ROUND = args.round  # global round
    cfg.OPTIM.MAX_EPOCH = 1  # local epoch
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step
    cfg.OPTIM.LR = args.lr  # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True




def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)
    # print_args(args, cfg)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    if args.model != "local":
        global_trainer = build_trainer(cfg)
        global_trainer.prompt_loss = PromptLoss(
            num_classes=cfg.DATASET.NUM_CLASSES,
            temperature=cfg.TRAINER.PROMPTFL.TEMPERATURE,
            alpha=cfg.TRAINER.PROMPTFL.ALPHA,
            beta=cfg.TRAINER.PROMPTFL.BETA,
            gamma=cfg.TRAINER.PROMPTFL.GAMMA,
            delta=cfg.TRAINER.PROMPTFL.DELTA,
            margin=cfg.TRAINER.PROMPTFL.MARGIN
        )
        global_trainer.class_priors = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        # global_trainer.global_estimator = EstimatorCV(feature_num=cfg.TRAINER.PROMPTFL.feat_dim, class_num=cfg.DATASET.NUM_CLASSES)

        print("global_trainer_isbuild_type:", type(global_trainer))
        # count_parameters(global_trainer.model,"prompt_learner")
        # count_parameters(global_trainer.model, "image_encoder")
        # count_parameters(global_trainer.model, "text_encoder")
        global_trainer.fed_before_train(is_global=True)

        # copy weights
        global_weights = global_trainer.model.state_dict()
    # local_weights, local_losses = [], []

    local_weights = [[] for i in range(args.num_users)]  # different
    local_weights_0 = [[] for i in range(args.num_users)]
    local_weights_1 = [[] for i in range(args.num_users)]
    local_weights_per = [{} for i in range(args.num_users)]
    local_proj = [{} for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()

    datanumber_client = []

    if args.trainer == 'CLIP':  # different
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()

    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()
    n_cls = len(local_trainer.dm.dataset.classnames)
    head_acc_list = []
    mid_acc_list = []
    tail_acc_list = []

    # local_coupling_params = []
    local_coupling_params = [[] for i in range(args.num_users)]

    # 绘图
    best_acc = 0
    best_epoch = 0
    last_class_accuracy = []
    best_per_class_accuracies = None
    per_class_accuracies = []
    best_class_accuracy = []

    # Initialize MAB schedulers
    mab = MABScheduler([1, 2, 3, 5, 7, 10])

    save_dir = os.path.join(args.output_dir, 'prompt_params')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("------------trainer == CLIP, global test_acpfl start-------------")
            # update global weights
            global_trainer.model.load_state_dict(global_weights)
            result = global_trainer.global_test(is_global=True, current_epoch=epoch)
            global_test_acc_list.append(result[0])
            global_test_error_list.append(result[1])
            global_test_f1_list.append(result[2])
            global_epoch_list.append(epoch)
            global_time_list.append(time.time() - start)
            last_class_accuracy = result[3]
            if result[0] > best_acc:
                best_acc = result[0]
                best_class_accuracy = last_class_accuracy
            print("------------global test_acpfl finish-------------")
            print("global_test_acc_list:", global_test_acc_list)
            print("maximum test_acpfl acc:", max(global_test_acc_list))
            print("mean of acc:", np.mean(global_test_acc_list[-5:]))
            print("std of acc:", np.std(global_test_acc_list[-5:]))


            print("Epoch on server :", epoch)
            break

        elif args.model == "cluster":
            if args.trainer == "CAPT":
                epoch_start_time = time.time()

                print(f"------------Epoch {epoch}: CAPT cluster training------------")

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print(f"Selected clients for this round: {idxs_users}")

                for idx in idxs_users:


                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)

                    local_coupling_param = local_trainer.model.coupling_function.state_dict()
                    local_coupling_params[idx] = copy.deepcopy(local_coupling_param)

                print("------------local train finish epoch:", epoch, "-------------")

                client_proportions = get_client_proportions(local_trainer.client_proportion, idxs_users,
                                                            cfg.DATASET.NUM_CLASSES)

                print("------------Client clustering start-------------")

                # MAB parameter selection
                intra_cluster_rounds = mab.get_value(mab.select_arm())
                similarity_iters = mab.get_value(mab.select_arm())
                dissimilarity_iters = mab.get_value(mab.select_arm())
                global_agg_freq = mab.get_value(mab.select_arm())

                print(f"MAB selected parameters: intra_cluster_rounds={intra_cluster_rounds}, "
                      f"similarity_iters={similarity_iters}, dissimilarity_iters={dissimilarity_iters}, "
                      f"global_agg_freq={global_agg_freq}")

                n_clusters_similarity = args.n_simclusters

                similarity_clusters = similarity_clustering(client_proportions, n_clusters_similarity)
                print("\nSimilarity Clustering:")
                print_cluster_results(similarity_clusters, idxs_users)

                for cluster in set(similarity_clusters):
                    cluster_members = [idxs_users[i] for i, c in enumerate(similarity_clusters) if c == cluster]
                    communicate_within_cluster_similarity(cluster_members, local_weights)

                n_clusters_dissimilarity = args.n_disclusters
                dissimilarity_clusters = dissimilarity_clustering(client_proportions, n_clusters_dissimilarity)
                print("\nDissimilarity Clustering:")
                print_cluster_results(dissimilarity_clusters, idxs_users)

                for cluster in set(dissimilarity_clusters):
                    cluster_members = [idxs_users[i] for i, c in enumerate(dissimilarity_clusters) if c == cluster]
                    communicate_within_cluster_dissimilarity(cluster_members, local_weights)

                print("------------Adaptive clustering and communication completed-------------")

                if epoch % global_agg_freq == 0:

                    global_class_aware_prompt = global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu()
                    aggregated_class_aware_prompt = aggregate_class_aware_prompts(client_proportions, local_weights,
                                                                                  idxs_users, cfg.DATASET.NUM_CLASSES,
                                                                                  global_class_aware_prompt)

                    global_trainer.model.prompt_learner.class_aware_ctx.data = aggregated_class_aware_prompt.to(
                        global_trainer.model.prompt_learner.class_aware_ctx.device)
                    for idx in idxs_users:
                        local_weights[idx]['prompt_learner.class_aware_ctx'] = aggregated_class_aware_prompt.to(
                            local_weights[idx]['prompt_learner.class_aware_ctx'].device)

                    global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                    global_trainer.model.load_state_dict(global_weights)

                    # update local
                    local_trainer.model.load_state_dict(global_weights, strict=False)

                    print("------------Global aggregation completed-------------")

                    print("------------global test start-------------")
                    result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                    prompt_state = {
                        'epoch': epoch,
                        'general_prompt': global_trainer.model.prompt_learner.general_ctx.detach().cpu(),
                        'class_aware_prompt': global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu()
                    }
                    torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                    accuracy, error, f1_score, last_class_accuracy = result[:4]
                    global_test_acc_list.append(result[0])
                    global_test_error_list.append(result[1])
                    global_test_f1_list.append(result[2])
                    global_epoch_list.append(epoch)
                    global_time_list.append(time.time() - start)

                    last_class_accuracy = result[3]
                    if result[0] > best_acc:
                        best_acc = result[0]
                        best_class_accuracy = last_class_accuracy

                        best_prompt_state = {
                            'epoch': epoch,
                            'general_prompt': global_trainer.model.prompt_learner.general_ctx.detach().cpu(),
                            'class_aware_prompt': global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu(),
                            'accuracy': best_acc,
                            'class_accuracy': best_class_accuracy
                        }
                        torch.save(best_prompt_state, os.path.join(save_dir, 'best_prompt_params.pth'))


                    print("------------global test finish-------------")
                    print("global_test_acc_list:", global_test_acc_list)
                    print("maximum test_acpfl acc:", max(global_test_acc_list))
                    print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                    print("std of acc:", np.std(global_test_acc_list[-5:]))
                    print(last_class_accuracy)
                    print(len(last_class_accuracy))

                    head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy,
                                                                                       local_trainer)
                    head_acc_list.append(head_acc)
                    mid_acc_list.append(medium_acc)
                    tail_acc_list.append(tail_acc)

                    print("Epoch on server :", epoch)

                    # Calculate reward and update MAB
                    convergence_rate = mab.calculate_convergence_rate(global_test_acc_list)
                    reward = mab.calculate_reward(accuracy, f1_score, convergence_rate)

                    mab.update(mab.get_arm_from_value(intra_cluster_rounds), reward, epoch)
                    mab.update(mab.get_arm_from_value(similarity_iters), reward, epoch)
                    mab.update(mab.get_arm_from_value(dissimilarity_iters), reward, epoch)
                    mab.update(mab.get_arm_from_value(global_agg_freq), reward, epoch)

                    print(f"Epoch {epoch}: Updated MAB schedulers with reward {reward}")


                else:
                    print(f"Skipping global aggregation at epoch {epoch}")

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

        elif args.model == "fedavg":
            if args.trainer == "PromptFL":
                print("use model == fedavg and trainer == PromptFL")
                m = max(int(args.frac * args.num_users), 1)  # different
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                # idxs_users = list(range(0,cfg.DATASET.USERS))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                # update global weights
                global_trainer.model.load_state_dict(global_weights)  # hsh

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                prompt_state = {
                    'epoch': epoch,
                    'ctx_prompt': global_trainer.model.prompt_learner.ctx.detach().cpu(),
                }
                torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]  # 获取类别准确度
                # 更新最佳准确度和对应的类别准确度
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                    best_prompt_state = {
                        'epoch': epoch,
                        'ctx_prompt': global_trainer.model.prompt_learner.ctx.detach().cpu(),
                        # 'class_aware_prompt': global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu(),
                        'accuracy': best_acc,
                        'class_accuracy': best_class_accuracy
                    }
                    torch.save(best_prompt_state, os.path.join(save_dir, 'best_prompt_params.pth'))



                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)

                print("Epoch on server :", epoch)

            elif args.trainer == "CAPT":
                print(f"------------Epoch {epoch}: CAPT training, fedavg server------------")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print(f"Selected clients for this round: {idxs_users}")
                print("------------local train start epoch:", epoch, "-------------")
                local_class_priors = []
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.prompt_loss = copy.deepcopy(global_trainer.prompt_loss)
                    local_trainer.class_priors = copy.deepcopy(global_trainer.class_priors)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                    local_class_priors.append(local_trainer.class_priors)
                print("------------local train finish epoch:", epoch, "-------------")

                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                # update global class priors
                global_trainer.class_priors = update_class_priors(global_trainer.class_priors, local_class_priors,
                                                                  idxs_users, datanumber_client)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)
                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)
                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy



                print("------------global test finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))
                head_acc, mid_acc, tail_acc = calculate_accuracy_5(last_class_accuracy)
                head_acc_list.append(head_acc)
                mid_acc_list.append(mid_acc)
                tail_acc_list.append(tail_acc)
                print("Epoch on server :", epoch)

            elif args.trainer == "MaPLe":
                print("use model == fedavg and trainer == MaPLe")
                m = max(int(args.frac * args.num_users), 1)  # different
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                # idxs_users = list(range(0,cfg.DATASET.USERS))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                # update global weights
                global_trainer.model.load_state_dict(global_weights)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)
                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))
                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                print("Epoch on server :", epoch)

            elif args.trainer == "CoCoOp":
                print("use model == fedavg and trainer == CoCoOp")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)


                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)

                print("Epoch on server :", epoch)

            elif args.trainer == "ClipLora":
                print("use model == fedavg and trainer == ClipLoRa")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)


                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)

                print("Epoch on server :", epoch)

            elif args.trainer == "KgCoOp":
                print("use model == fedavg and trainer == PromptFL")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                prompt_state = {
                    'epoch': epoch,
                    'ctx_prompt': global_trainer.model.prompt_learner.ctx.detach().cpu(),
                }
                torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)

                print("Epoch on server :", epoch)

            elif args.trainer == "FedClip":
                print("use model == fedavg and trainer == FedClip")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")

                local_weights = {}
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)

                print("------------local train finish epoch:", epoch, "-------------")

                # 只聚合img_adap的参数
                for key in global_weights.keys():
                    if 'img_adap' in key:
                        temp = torch.zeros_like(global_weights[key])
                        total_weight = sum([datanumber_client[idx] for idx in idxs_users])
                        for client_idx in idxs_users:
                            temp += (datanumber_client[client_idx] / total_weight) * local_weights[client_idx][key]
                        global_weights[key] = temp

                # 更新全局模型
                global_trainer.model.load_state_dict(global_weights)

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_5(last_class_accuracy, local_trainer)
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)

                print("Epoch on server :", epoch)



    print("------------Specific info-------------")

    print("maximum test_acpfl acc:", max(global_test_acc_list))
    print("mean of acc:", np.mean(global_test_acc_list[-5:]))
    print("std of acc:", np.std(global_test_acc_list[-5:]))

    print("Global Test Accuracy List:")
    print(global_test_acc_list)
    print("\nGlobal Test Error List:")
    print(global_test_error_list)
    print("\nGlobal Test F1 Score List:")
    print(global_test_f1_list)
    print("\nGlobal Epoch List:")
    print(global_epoch_list)
    print("\nGlobal Time List:")
    print(global_time_list)



    print("test_acpfl acc:", max(global_test_acc_list))
    print(best_class_accuracy)
    calculate_accuracy_5(best_class_accuracy,local_trainer)


    print("\nFinish!")

    if args.trainer != 'CLIP':
        for idx in idxs_users:
            local_trainer.fed_after_train()
    if args.model != 'local':
        global_trainer.fed_after_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cluster",help="model of aggregation, choose from:cluster,FedOTP(used with CAPT), fedavg, fedprox, local(The last three are used with PromptFL)")
    parser.add_argument("--trainer", type=str, default="CAPT",help="name of trainer, choose from: CAPT,CLIP, PromptFL, CAPT,MaPLe,CoOp")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.4, help='the fraction of clients: C')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test_acpfl batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # parameters of datasets
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir')
    parser.add_argument('--beta', type=float, default=0.05,help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor，IF = 100, 50 and 10')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=1, help="number of prompts")  # 2
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to aggregate")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")
    parser.add_argument('--csc', default="True", help="is using the ctx init, set True for CLIP")


    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--imagenetroot", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/test/", help="output directory")
    parser.add_argument("--config-file", type=str, default="configs/trainers/CAPT/vit_b16.yaml",help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar10_LT.yaml",help="path to config file for dataset setup")  #############
    parser.add_argument("--resume", type=str, default=None,help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,help="modify config options using the command-line")

    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float, metavar='LR', help='initial learning rate',dest='lr')
    parser.add_argument('--schedule', default=[6, 10], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset', default="imagenet_LT")
    parser.add_argument('--visualize_interval', type=int, default=10, help="Interval for generating visualizations")
    parser.add_argument('--n_general', type=int, default=1, help="number of text encoder of text prompts")
    parser.add_argument('--n_disclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--n_simclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--prompt_depth', type=int, default=9)
    # LoRA arguments
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')

    args = parser.parse_args()
    main(args)








