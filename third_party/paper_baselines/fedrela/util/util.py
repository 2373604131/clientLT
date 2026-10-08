import copy

import numpy as np


def shot_split(class_dtribution, threshold_3shot=(75, 95)):
    """Split class IDs into head / middle / tail by cumulative sample proportion."""
    threshold_3shot = list(threshold_3shot)
    class_distribution = copy.deepcopy(class_dtribution)
    map_num2classid2accumu = [[], [], []]
    for classid in range(len(class_dtribution)):
        map_num2classid2accumu[0].append(class_distribution[classid])
        map_num2classid2accumu[1].append(classid)
    for j in range(len(map_num2classid2accumu[0]) - 1):
        for i in range(len(map_num2classid2accumu[0]) - j - 1):
            if map_num2classid2accumu[0][i] < map_num2classid2accumu[0][i + 1]:
                map_num2classid2accumu[0][i], map_num2classid2accumu[0][i + 1] = (
                    map_num2classid2accumu[0][i + 1],
                    map_num2classid2accumu[0][i],
                )
                map_num2classid2accumu[1][i], map_num2classid2accumu[1][i + 1] = (
                    map_num2classid2accumu[1][i + 1],
                    map_num2classid2accumu[1][i],
                )
    map_num2classid2accumu[2] = (
        np.cumsum(np.array(map_num2classid2accumu[0])) / sum(map_num2classid2accumu[0]) * 100
    ).tolist()

    three_shot_dict = {"head": [], "middle": [], "tail": []}
    cut1 = cut2 = 0
    accu_range_auxi = [0] + map_num2classid2accumu[2]
    accu_range = copy.deepcopy(accu_range_auxi)
    for i in range(1, len(accu_range)):
        accu_range[i] = [accu_range_auxi[i - 1], accu_range_auxi[i]]
    del accu_range[0]
    for i, r in enumerate(accu_range):
        if threshold_3shot[0] > r[0] and threshold_3shot[0] <= r[1]:
            cut1 = i
        if threshold_3shot[1] > r[0] and threshold_3shot[1] <= r[1]:
            cut2 = i

    for i in range(len(map_num2classid2accumu[1])):
        cid = map_num2classid2accumu[1][i]
        if i <= cut1:
            three_shot_dict["head"].append(cid)
        elif i <= cut2:
            three_shot_dict["middle"].append(cid)
        else:
            three_shot_dict["tail"].append(cid)
    return three_shot_dict, map_num2classid2accumu
