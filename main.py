import os
import time
import warnings
from model import *
from data_processing import data_param_prepare
from config import parse_args_lastfm
import torch.utils.data as data
from evaluation import *

warnings.filterwarnings('ignore')


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_func(data_train, metapath, metapath_T, feature, train_loader, data_loader, valid_node, test_node, valid_mask,
               test_mask, valid_ground_truth_list,
               test_ground_truth_list, u_iid_list):
    model = Dual(args, data_train, metapath, metapath_T, feature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_recall_tuple = 0
    early_stop_count = 0
    u_iid_list = torch.LongTensor(u_iid_list).to(args.device)
    all_start_time = time.time()
    epoch = 0
    while True:
        epoch += 1
        model.train()
        start_time = time.time()
        for batch, nodes in enumerate(train_loader):
            model.zero_grad()
            loss = model(nodes, u_iid_list)
            loss.backward()
            optimizer.step()

        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

        Recall, NDCG = test_func(model, valid_node, valid_ground_truth_list, valid_mask, args.top_k, data_loader)
        print('Valid Results:')
        for k in args.top_k:
            print("Top{:d} \t Recall: {:.4f}\tNDCG: {:.4f}\t".format(k, Recall[k], NDCG[k]))
        test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        all_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - all_start_time))
        print(
            'The time for epoch {} is: Time-consuming = {}, train time = {}, test time = {}'.format(epoch, all_time,
                                                                                                    train_time,
                                                                                                    test_time))
        if Recall[args.top_k[-1]] > best_recall_tuple:
            best_recall_tuple = Recall[args.top_k[-1]]
            early_stop_count = 0
            torch.save(model.state_dict(), args.model_save_path)
            print(f'Saving current best:{args.model_save_path}')
        else:
            early_stop_count += 1
        print(f'epoch:{epoch}\tearly_stop_count:{early_stop_count}')
        if early_stop_count >= args.patience:
            print('####################################################################################')
            all_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - all_start_time))
            print('Early stop is triggered at {} epochs. Time-consuming = {}'.format(epoch, all_time))
            break
        print('####################################################################################')
    model.load_state_dict(torch.load(args.model_save_path))
    print(f'Loading model structure and parameters from :{args.model_save_path}')
    Recall, NDCG = test_func(model, test_node, test_ground_truth_list, test_mask, args.top_k, data_loader)
    print('Test Results:')
    for k in args.top_k:
        print("Top{:d} \t Recall: {:.4f}\tNDCG: {:.4f}\t".format(k, Recall[k], NDCG[k]))


def test_func(model, test_node, test_ground_truth_list, mask, topk, test_loader):
    with torch.no_grad():
        model.eval()
        Recall, NDCG = dict(), dict()
        rating_all = list()
        for batch, nodes in enumerate(test_loader):
            rating_all.append(model.test_foward(nodes))
        rating_all = torch.cat(rating_all, dim=0)
        rating_all = rating_all.cpu()
        rating_all += mask
        for k in topk:
            _, rating_list_all = torch.topk(rating_all, k=k)
            rat = rating_list_all.numpy()
            rat = rat[test_node]
            groudtrue = [test_ground_truth_list[u] for u in test_node]
            precision, recall, ndcg = test_one_batch(rat, groudtrue, k)
            Recall[k] = recall / len(test_node)
            NDCG[k] = ndcg / len(test_node)

    return Recall, NDCG


if __name__ == '__main__':

    args = parse_args_lastfm()
    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = int(np.argmax(memory_available))
        args.device = 'cuda:{}'.format(gpu_id)
    else:
        args.device = 'cpu'

    setup_seed(args.seed)
    args.train_time = str(time.strftime('%m-%d-%H-%M'))
    args.model_save_path = f'./save/{args.dataname}/{args.train_time}.pt'

    print(args)
    print('-----------------------------------------------')
    print('##################################################################################')

    data_train, metapath, metapath_T, feature, valid, test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list, u_iid_list = data_param_prepare(
        args)
    train_loader = data.DataLoader(range(args.n_user), batch_size=192, shuffle=True)
    data_loader = data.DataLoader(range(args.n_user), batch_size=192, shuffle=False)
    train_func(data_train, metapath, metapath_T, feature, train_loader, data_loader, valid, test, valid_mask, test_mask,
               valid_ground_truth_list, test_ground_truth_list, u_iid_list)
    print('##################################################################################')
    print('test over !!!')
