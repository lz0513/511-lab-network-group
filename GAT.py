import time
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from GNN import load_data, normalize_adj

# 实现计算准确性的函数
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), \
                             h.repeat(1, 1, N).view(N * N, -1)], dim=1) \
                            .view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

def main():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TensorBoard
    writer = SummaryWriter(log_dir='./logs')

    # 1. 加载Cora数据集
    dataset, class_num = load_data()
    
    # 处理邻接矩阵为稀疏格式
    adj = to_scipy_sparse_matrix(dataset.edge_index)
    adj = normalize_adj(adj)
    adj = torch.FloatTensor(adj.toarray()).to(device)

    dataset = dataset.to(device)

    # 2. 划分训练集、验证集和测试集
    train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask
    train_label, val_label, test_label = dataset.y[train_mask], dataset.y[val_mask], dataset.y[test_mask]
    train_idx, val_idx, test_idx = torch.nonzero(train_mask).squeeze(), torch.nonzero(val_mask).squeeze(), torch.nonzero(test_mask).squeeze()

    # 3. 初始化模型
    model = GAT(nfeat=dataset.num_features,
                nhid=8,
                nclass=class_num,
                dropout=0.5,
                alpha=0.2,
                nheads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 4. 训练模型
    for epoch in range(200):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(dataset.x, adj)
        loss_train = F.nll_loss(output[train_idx], train_label)
        acc_train = accuracy(output[train_idx], train_label)
        loss_train.backward()
        optimizer.step()

        # 5. 验证模型
        model.eval()
        with torch.no_grad():
            output = model(dataset.x, adj)
            loss_val = F.nll_loss(output[val_idx], val_label)
            acc_val = accuracy(output[val_idx], val_label)

        # 6. 记录日志
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        writer.add_scalar('loss_train', loss_train.item(), epoch + 1)
        writer.add_scalar('acc_train', acc_train.item(), epoch + 1)
        writer.add_scalar('loss_val', loss_val.item(), epoch + 1)
        writer.add_scalar('acc_val', acc_val.item(), epoch + 1)

    # 7. 测试模型
    model.eval()
    with torch.no_grad():
        output = model(dataset.x, adj)
        loss_test = F.nll_loss(output[test_idx], test_label)
        acc_test = accuracy(output[test_idx], test_label)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        writer.add_scalar('loss_test', loss_test.item(), epoch + 1)
        writer.add_scalar('acc_test', acc_test.item(), epoch + 1)
    writer.close()

if __name__ == '__main__':
    main()