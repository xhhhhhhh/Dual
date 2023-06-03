import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import scipy.sparse as sp


class Gate(nn.Module):
    def __init__(self, in_size):
        super(Gate, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.Tanh(),
            nn.Linear(in_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        betas = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)        ###

        return (betas * z).sum(1)  # (N, D * K)      ###


class Dual(nn.Module):

    def __init__(self, args, data_train,metapath, metapath_T, feature):
        super(Dual, self).__init__()

        self.args = args
        self.pr_lable = torch.FloatTensor(metapath.A).to(args.device)
        graph = self.get_graph(data_train)
        metapath_graph = self.get_graph(metapath, metapath_T)
        self.graph = self.get_norm_adj_mat(graph).to(args.device)
        self.metapath_graph = self.get_norm_adj_mat(metapath_graph>0).to(args.device)
        self.n_user, self.n_item = args.n_user, args.n_item
        self.feature = torch.tensor(feature.A).to(torch.float32).to(args.device)
        self.pos = torch.FloatTensor(metapath_graph.A==1).to(args.device)
        self.neg = torch.FloatTensor(metapath_graph.A==0).to(args.device)

        self.embeds_1 = torch.nn.Embedding(num_embeddings=args.n_node, embedding_dim=args.embedding_dim)
        self.embeds_2 = torch.nn.Embedding(num_embeddings=args.n_node, embedding_dim=args.embedding_dim)

        self.embeds_item_last = torch.zeros(1, args.embedding_dim).to(args.device)
        self.attention_user_re = Gate(in_size=args.embedding_dim)
        self.attention_item_re = Gate(in_size=args.embedding_dim)
        self.attention_user_pr = Gate(in_size=args.embedding_dim)
        self.attention_item_pr = Gate(in_size=args.embedding_dim)

        self.r_re = nn.Parameter(torch.randn(args.embedding_dim, 1))
        self.r_pr = nn.Parameter(torch.randn(args.embedding_dim, 1))
        self.τ = args.τ
        self.MSELoss = nn.MSELoss(reduction='sum')

        self.apply(xavier_uniform_initialization)

    def get_graph(self, matrix1, matrix2=None):
        uu, ii = sp.eye(self.args.n_user, self.args.n_user), sp.eye(self.args.n_item, self.args.n_item)

        graph_top = sp.hstack((uu, matrix1))
        if matrix2 is None:
            graph_bottom = sp.hstack((matrix1.T, ii))
        else:
            graph_bottom = sp.hstack((matrix2, ii))
        graph = sp.vstack((graph_top, graph_bottom))

        return graph

    def get_embedding(self, user_lists, item_lists):
        semantic_user = torch.stack(user_lists[0], dim=1)
        semantic_item = torch.stack(item_lists[0], dim=1)
        h_user_re = self.attention_user_re(semantic_user)
        h_item_re = self.attention_item_re(semantic_item)

        semantic_user = torch.stack(user_lists[1], dim=1)
        semantic_item = torch.stack(item_lists[1], dim=1)
        h_user_pr = self.attention_user_pr(semantic_user)
        h_item_pr = self.attention_item_pr(semantic_item)

        return h_user_re, h_item_re, h_user_pr, h_item_pr


    def forward(self, nodes, u_iid_list):
        h_user_re, h_item_re, h_user_pr, h_item_pr = self.conv()
        loss = self.rec_loss(h_user_re[nodes], h_item_re, u_iid_list[nodes], self.r_re, self.args.neg_weight)
        pq_pr = h_user_pr[nodes].unsqueeze(1) * h_item_pr
        hpq_pr = pq_pr.matmul(self.r_pr).squeeze(2)
        rating_pr = self.pr_lable[nodes]
        loss += self.args.pr_w * self.MSELoss(hpq_pr, rating_pr)
        loss += self.args.con_w * self.con_loss(h_user_re, h_item_re, h_user_pr, h_item_pr)
        return loss


    def rec_loss(self, h_u, h_i, pos_iids, r, neg_weight, rating=None):

        h_i = torch.cat((h_i, self.embeds_item_last.data), 0)
        item_num = h_i.shape[0] - 1
        mask = (~(pos_iids.eq(item_num))).float()
        pos_embs = h_i[pos_iids]
        pos_embs = pos_embs * mask.unsqueeze(2)
        pq = h_u.unsqueeze(1) * pos_embs
        hpq = pq.matmul(r).squeeze(2)
        if rating == None:
            pos_data_loss = torch.sum((1 - neg_weight) * hpq.square() - 2.0 * hpq)
        else:
            pos_data_loss = torch.sum((1 - neg_weight) * hpq.square() - 2.0 * hpq * rating)
        part_1 = h_u.unsqueeze(2).bmm(h_u.unsqueeze(1))
        part_2 = h_i.unsqueeze(2).bmm(h_i.unsqueeze(1))

        part_1 = part_1.sum(0)
        part_2 = part_2.sum(0)
        part_3 = r.mm(r.t())

        all_data_loss = torch.sum(part_1 * part_2 * part_3)
        loss = neg_weight * all_data_loss + pos_data_loss

        return loss

    def con_loss(self, h_user_re, h_item_re, h_user_pr, h_item_pr):
        emb_re = torch.cat([h_user_re, h_item_re], dim=0)
        emb_pr = torch.cat([h_user_pr, h_item_pr], dim=0)

        emb_re = torch.nn.functional.normalize(emb_re, p=2, dim=-1)
        emb_pr = torch.nn.functional.normalize(emb_pr, p=2, dim=-1)

        node_embeds = emb_re
        con_ttl_score = torch.mm(node_embeds, emb_pr.T)

        con_pos_score = (self.pos * torch.exp(con_ttl_score / self.τ)).sum(-1)
        con_ttl_score = (self.neg * torch.exp(con_ttl_score / self.τ)).sum(-1)
        loss = -torch.log(con_pos_score / con_ttl_score)

        return loss.sum()

    def conv(self):
        embeds_1 = torch.matmul(self.feature, self.embeds_1.weight)
        embeds_2 = torch.matmul(self.feature, self.embeds_2.weight)

        input_1 = self.graphconv(self.args.layers, self.graph, embeds_1)
        input_2 = self.graphconv(self.args.layers, self.graph, embeds_2)
        input_3 = self.graphconv(self.args.layers, self.metapath_graph, embeds_2)
        input_4 = self.graphconv(self.args.layers, self.metapath_graph, embeds_1)

        user_emb_1, item_emb_1 = torch.split(input_1, [self.args.n_user, self.args.n_item])
        user_emb_2, item_emb_2 = torch.split(input_2, [self.args.n_user, self.args.n_item])
        user_emb_3, item_emb_3 = torch.split(input_3, [self.args.n_user, self.args.n_item])
        user_emb_4, item_emb_4 = torch.split(input_4, [self.args.n_user, self.args.n_item])

        user_lists = [[user_emb_1, user_emb_2, user_emb_3], [user_emb_2, user_emb_3, user_emb_4]]
        item_lists = [[item_emb_1, item_emb_2, item_emb_3], [item_emb_2, item_emb_3, item_emb_4]]
        h_user_re, h_item_re, h_user_pr, h_item_pr = self.get_embedding(user_lists,item_lists)
        return h_user_re, h_item_re, h_user_pr, h_item_pr

    def graphconv(self,layers, graph, embeding):
        for layer_idx in range(layers):
            embeding = torch.sparse.mm(graph, embeding)
        return embeding

    def get_norm_adj_mat(self, A):
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def test_foward(self,nodes):

        h_user_re, h_item_re, h_user_pr, h_item_pr = self.conv()
        scores_re = (h_user_re[nodes].unsqueeze(1) * h_item_re).matmul(self.r_re).squeeze(2)
        return scores_re


def xavier_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)