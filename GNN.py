import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.utils.tensorboard import SummaryWriter

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
        
    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
def load_data():
    dataset = Planetoid(root='./data/Cora', name='Cora')
    class_num = dataset.num_classes
    dataset = dataset[0]
    return dataset, class_num

def normalize_adj(mx):
    mx = mx + sp.eye(mx.shape[0]) # A = A + I
    rowsum = np.array(mx.sum(1))  
    r_inv = np.power(rowsum, -1/2).flatten() # degree matrix D 
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv) #得到最终标准化的邻接矩阵
    return mx

def main():
    # 动态设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TensorBoard
    writer = SummaryWriter(log_dir='./logs')

    # 1. 加载Cora数据集
    dataset, class_num = load_data()
    adj = to_scipy_sparse_matrix(dataset.edge_index)
    # 2. 标准化邻接矩阵
    adj_norm = normalize_adj(adj)
    dataset = dataset.to(device)
    adj_norm = torch.FloatTensor(adj_norm.toarray()).to(device)

    # 2. 构建图卷积神经网络模型
    model = GCN(dataset.num_features, 128, class_num).to(device)

    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 3. 训练模型
    for epoch in range(200):
        model.train()
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(dataset.x, adj_norm)
        # 计算损失
        loss = criterion(output[dataset.train_mask], dataset.y[dataset.train_mask])
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        
        # 记录训练损失到 TensorBoard
        writer.add_scalar('Training Loss', loss.item(), global_step=epoch)
        
        # 记录学习率到 TensorBoard
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        
        # 记录权重直方图到 TensorBoard
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)
        
        # 打印训练信息
        print('Epoch: {:04d}, Loss: {:.4f}'.format(epoch + 1, loss.item()))

        # 记录验证集损失
        if dataset.val_mask is not None:
            val_output = model(dataset.x, adj_norm)
            val_loss = criterion(val_output[dataset.val_mask], dataset.y[dataset.val_mask])
            writer.add_scalar('Validation Loss', val_loss.item(), global_step=epoch)

    # 4. 测试模型
    model.eval()
    _, pred = model(dataset.x, adj_norm).max(dim=1)
    correct = float(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
    acc = correct / dataset.test_mask.sum().item()
    print('Test Accuracy: {:.4f}'.format(acc))

    # 记录测试准确率到 TensorBoard
    writer.add_scalar('Test Accuracy', acc, global_step=200)

    # 关闭 TensorBoard
    writer.close()

if __name__ == '__main__':
    main()