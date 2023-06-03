import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict

def data_param_prepare(args):
    ui_data = pd.read_csv(f'./data/{args.dataname}/ua.txt', sep='\t', header=None,
                          names=['user_id', 'item_id'])
    ui_data['user_id'] = ui_data['user_id'].astype('category')
    ui_data['item_id'] = ui_data['item_id'].astype('category')
    user_id = ui_data['user_id'].cat.codes.values
    item_id = ui_data['item_id'].cat.codes.values
    user_num = user_id.max() + 1
    item_num = item_id.max() + 1

    args.n_user, args.n_item = user_num, item_num
    args.n_node = args.n_user + args.n_item

    link = np.ones(len(ui_data['user_id']))
    data_train = csr_matrix((link, (user_id, item_id)), shape=(user_num, item_num))
    data_valid = csr_matrix((link, (user_id, item_id)), shape=(user_num, item_num))
    data_test = csr_matrix((link, (user_id, item_id)), shape=(user_num, item_num))

    ii_data = pd.read_csv(f'./data/{args.dataname}/ii.txt', sep='\t', header=None,
                          names=['item_id', 'friend_id'])
    ii_data['item_id'] = ii_data['item_id'].astype('category')
    ii_data['friend_id'] = ii_data['friend_id'].astype('category')
    link = np.ones(len(ii_data['item_id']))
    item_id = ii_data['item_id'].cat.codes.values
    friend_id = ii_data['friend_id'].cat.codes.values
    II = csr_matrix((link, (item_id, friend_id)), shape=(item_num, item_num))

    uu_data = pd.read_csv(f'./data/{args.dataname}/uu.txt', sep='\t', header=None,
                          names=['user_id', 'friend_id'])
    uu_data['user_id'] = uu_data['user_id'].astype('category')
    uu_data['friend_id'] = uu_data['friend_id'].astype('category')
    link = np.ones(len(uu_data['user_id']))
    user_id = uu_data['user_id'].cat.codes.values
    friend_id = uu_data['friend_id'].cat.codes.values
    UU = csr_matrix((link, (user_id, friend_id)), shape=(user_num, user_num))

    float_mask = np.random.permutation(np.linspace(0, 1, len(ui_data['user_id'])))
    data_train.data[float_mask >= args.split_ratio['train']] = 0
    data_valid.data[
        (float_mask < args.split_ratio['train']) | (float_mask > (1 - args.split_ratio['test']))] = 0
    data_test.data[float_mask <= (1 - args.split_ratio['test'])] = 0

    link = np.ones(len(data_train.nonzero()[0]))
    data_train = csr_matrix((link, data_train.nonzero()), shape=(user_num, item_num))

    link = np.ones(len(data_valid.nonzero()[0]))
    data_valid = csr_matrix((link, data_valid.nonzero()), shape=(user_num, item_num))

    link = np.ones(len(data_test.nonzero()[0]))
    data_test = csr_matrix((link, data_test.nonzero()), shape=(user_num, item_num))

    UUI = UU * data_train
    IUU = normalize(UUI.T, axis=1, norm='max')
    UUI = normalize(UUI, axis=1, norm='max')

    UUI = csr_matrix(np.around(UUI.A, 1))
    IUU = csr_matrix(np.around(IUU.A, 1))

    user_feature = sp.hstack((UU, data_train))
    item_feature = sp.hstack((data_train.T, II))
    feature = sp.vstack((user_feature,item_feature))
    valid, test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list = train_test_split(
        data_train,
        data_valid,
        data_test)

    u_train, i_train = data_train.nonzero()

    u_iid_dict = defaultdict(list)
    for i in range(len(u_train)):
        u_iid_dict[u_train[i]].append(i_train[i])

    u_max_i = max([len(i) for i in u_iid_dict.values()])
    for i, id_list in u_iid_dict.items():
        if len(id_list) < u_max_i:
            u_iid_dict[i] += [item_num] * (u_max_i - len(id_list))

    u_id_train = np.array(list(set(u_train)), dtype=np.int32)
    u_iid_list = []
    for i in range(len(u_id_train)):
        u_iid_list.append(u_iid_dict[i])

    return data_train,UUI, IUU, feature, valid, test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list, u_iid_list

def train_test_split(UI_train, UI_valid, UI_test):
    n_head, n_tail = UI_train.shape
    u_train, i_train = UI_train.nonzero()
    u_valid, i_valid = UI_valid.nonzero()
    u_test, i_test = UI_test.nonzero()

    # list2txt('weeplaces_train.csv', u_train, i_train, delimiter=',')
    # list2txt('weeplaces_valid.csv', u_valid, i_valid, delimiter=',')
    # list2txt('weeplaces_test.csv', u_test, i_test, delimiter=',')

    train_data = np.column_stack((u_train, i_train))
    valid_data = np.column_stack((u_valid, i_valid))
    test_data = np.column_stack((u_test, i_test))

    u_valid = np.array(list(set(u_valid)), dtype='int64')
    u_test = np.array(list(set(u_test)), dtype='int64')

    valid_mask = torch.zeros(n_head, n_tail)

    for (u, i) in train_data:
        valid_mask[u][i] = -np.inf

    test_mask = valid_mask.clone()
    valid_ground_truth_list = [[] for _ in range(n_head)]
    for (u, i) in valid_data:
        valid_ground_truth_list[u].append(i)
        test_mask[u][i] = -np.inf

    test_ground_truth_list = [[] for _ in range(n_head)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    return u_valid, u_test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list