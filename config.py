import argparse
def parse_args_lastfm():
    parser = argparse.ArgumentParser(description='Dual')
    parser.add_argument('--dataname', type=str, default='last_fm', help='Name of dataset.')
    parser.add_argument("--embedding_dim", type=int, default=256, help='Hidden layer dim.')
    parser.add_argument("--seed", type=int, default=100, help='seed')
    parser.add_argument('--split_ratio', type=dict, default={'train': 0.8, 'test': 0.1}, help='split_ratio')
    parser.add_argument('--τ', type=float, default=0.2, help='item_temp')
    parser.add_argument('--top_k', type=list, default=[5, 10, 15, 20], help='top_k')
    parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--layers', type=int, default=2, help='layers')
    parser.add_argument('--pr_w', type=float, default=0.3, help='predictive task weight')
    parser.add_argument('--con_w', type=float, default=5e-4, help='contrastive task weight')
    parser.add_argument('--neg_weight', type=float, default=0.15, help='negative sample weight')

    return parser.parse_args()
