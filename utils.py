import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj, get_laplacian, degree
import random
import os

#设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

#生成器类
class Generator(nn.Module):

    def __init__(self, noise_dim, input_dim, output_dim, dropout):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(output_dim, output_dim)

        hid_layers = []
        dims = [noise_dim + output_dim, 64, 128, 256]
        for i in range(len(dims) - 1):
            d_in = dims[i]
            d_out = dims[i + 1]
            hid_layers.append(nn.Linear(d_in, d_out))
            #与Ghost不同的地方(添加了一个批归一化层)
            # hid_layers.append(nn.BatchNorm1d(d_out))  # 添加BN层
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p = dropout, inplace = False))
        self.hid_layers = nn.Sequential(* hid_layers)
        self.nodes_layer = nn.Linear(256, input_dim)
    
    def forward(self, z, c):
        #标签嵌入 
        label_emb = self.emb_layer.forward(c)    
        #拼接噪声和标签嵌入
        z_c = torch.cat((label_emb, z), dim = -1)
        #通过隐藏层
        hid = self.hid_layers(z_c)
        #生成最终节点特征
        node_logits = self.nodes_layer(hid)
        return node_logits

# 链接预测器类，基于两层MLP架构
class LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super(LinkPredictor, self).__init__()
        # 第一层：将两个节点特征连接后映射到隐藏维度
        self.layer1 = nn.Linear(2 * input_dim, hidden_dim)  # 2倍输入维度，因为我们连接两个节点的特征
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        # 第二层(输出)：将隐藏表示映射到单一输出，表示两节点之间存在连接的概率
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_i, x_j):
        # 将两个节点的特征连接起来
        x = torch.cat([x_i, x_j], dim=-1)
        # 通过第一层
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # 通过第二层，得到预测的边存在概率
        x = self.layer2(x)
        # 应用sigmoid函数，得到0~1之间的概率
        x = self.sigmoid(x)
        return x
    
    def predict_links(self, node_features, threshold=0.5):
        """
        基于节点特征预测图的邻接矩阵
        
        参数:
        - node_features: 节点特征矩阵，形状为[num_nodes, feature_dim]
        - threshold: 预测边存在的概率阈值
        
        返回:
        - adj_matrix: 预测的邻接矩阵，形状为[num_nodes, num_nodes]
        """
        num_nodes = node_features.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=node_features.device)
        
        # 对每一对节点进行预测
        for i in range(num_nodes):
            # 获取当前节点特征并扩展维度以便广播
            x_i = node_features[i].unsqueeze(0).expand(num_nodes, -1)
            # 获取所有其他节点的特征
            x_j = node_features
            # 预测链接概率
            probs = self.forward(x_i, x_j).squeeze()
            # 根据阈值确定边的存在性
            edges = (probs > threshold).float()
            # 更新邻接矩阵
            adj_matrix[i] = edges
            
        return adj_matrix
    
#根据生成器生成的节点特征和相似度进行图的构建
def construct_graph(node_logits, link_predictor=None, threshold=0.5):
    """
    使用生成的节点特征和链接预测器构建图
    
    参数:
    node_logits: 生成的节点伪特征矩阵
    link_predictor: 链接预测器模型，如果为None则使用余弦相似度
    threshold: 确定边存在的概率阈值
    
    返回:
    adjacency_matrix: 构建的邻接矩阵
    """
    # 如果提供了链接预测器，使用它来预测边
    if link_predictor is not None:
        adjacency_matrix = link_predictor.predict_links(node_logits, threshold)
    else:
        # 使用余弦相似度计算节点间相似度
        node_features_norm = torch.nn.functional.normalize(node_logits, p=2, dim=1)
        adj_logits = torch.mm(node_features_norm, node_features_norm.t())
        # 根据阈值确定边的存在性
        adjacency_matrix = (adj_logits > threshold).float()
    
    # 移除自环（对角线元素置零）
    adjacency_matrix.fill_diagonal_(0)
    
    return adjacency_matrix

#深度特征正则化的前向钩子类
class DeepInversionHook:
    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # nch = input[0].shape[1]     #输入特征的通道数
        # mean = input[0].mean([0, 2, 3])     # 计算当前批次特征图的均值（沿批次、高度和宽度维度）
        # var = input[0].permute(1, 0, 2, 3).contigous().view([nch, -1]).var(1, unbiased = False)     #计算每个通道的方差

        # input[0] shape: [batch_size, features]
        mean = input[0].mean(0)
        var = input[0].var(0, unbiased = False)

        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else: #动量平滑后的均值和方差
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data) 

    def remove(self):
        self.hook.remove()   
 
#低频蒸馏损失函数，传入的输出应该是经过softmax层后
def edge_distribution_low(edge_idx, student_out, teacher_out):
    src = edge_idx[0]
    dst = edge_idx[1]
    print(f"edge_idx:{edge_idx}")
    print(f"src:{src}")
    print(f"dst:{dst}")
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)

    loss = criterion(student_out[src], teacher_out[dst])
    
    print(f"low loss requires_grad:{loss.requires_grad}")
    return loss

#Average Accuary 平均准确率
def AA(M_acc, T = None):        
    """
    M_acc[i, j] 第i个任务训练完成后, 在第 j 个任务上的准确率
    """
    if T is None:
        T = M_acc.size(0)
    ret = 0
    for i in range(0, T):
        ret += M_acc[T - 1, i]       #训练了T个任务，最后一个任务的编号是 T - 1
    ret /= T
    return ret

#Average Forgetting 平均遗忘率
def AF(M_acc, T = None):
    if T is None:
        T = M_acc.size(0)
    if T == 1:                  #第一个任务
        return -1
    ret = 0
    for i in range(0, T - 1):
        forgetting = M_acc[i, i] - M_acc[T - 1, i]
        ret += forgetting
    ret /= T - 1
    return ret

#计算图的拉普拉斯能量分布（LED）
def compute_led(graph_data):
    """
    参数:
    graph_data: 图数据对象 Data(x, edge_index, y)
    返回:
    energy_distribution: 拉普拉斯能量分布 [N,]
    """
    nodes_feature = graph_data.x  # [N, d] 节点特征矩阵
    edge_index = graph_data.edge_index
    num_nodes = nodes_feature.shape[0]
        
    # 计算标准化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization='sym'  # 对称标准化
    )
        
    # 转换为稠密矩阵
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    # 特征值分解，获取特征向量矩阵 U
    eigenvalues, eigenvectors = torch.linalg.eigh(L)  # U: [N, N]
    U = eigenvectors  # 特征向量矩阵
    # 计算图傅里叶变换 \hat{X} = U^T X
    X_hat = torch.matmul(U.T, nodes_feature)  # [N, d] 傅里叶变换后的特征
    # 计算每个频率分量的能量（所有特征维度的平方和）
    energy_per_freq = torch.sum(X_hat ** 2, dim=1)  # [N,] 每个频率的能量
    # 计算总能量
    total_energy = torch.sum(energy_per_freq)
        
    # 计算能量分布（归一化）  \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
    if total_energy > 0:
        energy_distribution = energy_per_freq / total_energy  # [N,] 归一化的能量分布
    else:
        energy_distribution = torch.zeros_like(energy_per_freq)
    
    return energy_distribution
    

#高斯核函数
def gaussian_kernel1(x):        #np
    return (1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2))

def gaussian_kernel(x):
    return (1 / torch.sqrt(torch.tensor(2 * np.pi, device=x.device))) * torch.exp(-0.5 * x ** 2)


#核密度估计函数
def KDE(x, y, bandwidth, kernel_func):
    n = y.shape[0]
    kernel_values = kernel_func((x - y) / bandwidth)
    density_estimation = torch.sum(kernel_values) / (n * bandwidth)
    return density_estimation


def apply_kde_to_energy_dist(energy_dist, bandwidth, eval_nums, device):
    # energy_dist: [N,] torch tensor
    x_eval = torch.linspace(0, 1, eval_nums, device=device)
    prob_dist = torch.stack([KDE(xi, energy_dist, bandwidth, gaussian_kernel) for xi in x_eval])
    prob_dist = prob_dist / torch.sum(prob_dist)
    return prob_dist

def compute_js_divergence_from_prob_dist(prob_dist_1, prob_dist_2):
    M = 0.5 * (prob_dist_1 + prob_dist_2)
    epsilon = 1e-10
    prob_dist_1 = prob_dist_1 + epsilon
    prob_dist_2 = prob_dist_2 + epsilon
    M = M + epsilon
    kl_1_m = torch.sum(prob_dist_1 * torch.log(prob_dist_1 / M))
    kl_2_m = torch.sum(prob_dist_2 * torch.log(prob_dist_2 / M))
    js_divergence = 0.5 * kl_1_m + 0.5 * kl_2_m
    return js_divergence


#得到生成图与客户端图之间的SC值
def get_SC(synthetic_data, clients_nodes_num, clients_graph_energy, device, h):
    synthetic_energy_dist = compute_led(synthetic_data)  # torch tensor
    max_client_nodes = max(clients_nodes_num)
    synthetic_nodes_num = synthetic_data.x.shape[0]
    max_nodes_num = max(max_client_nodes, synthetic_nodes_num)
    synthetic_prob_dist = apply_kde_to_energy_dist(synthetic_energy_dist, h, max_nodes_num, device)

    sc_values = []
    total_nodes = sum(clients_nodes_num)
    for i, client_energy_dist in enumerate(clients_graph_energy):
        client_prob_dist = apply_kde_to_energy_dist(client_energy_dist, h, max_nodes_num, device)
        js_divergence = compute_js_divergence_from_prob_dist(synthetic_prob_dist, client_prob_dist)
        sc_values.append(js_divergence)
    weighted_sc = torch.tensor(0.0, device=device)
    for i, sc_value in enumerate(sc_values):
        weight = clients_nodes_num[i] / total_nodes
        weighted_sc += weight * sc_value
    return weighted_sc  # 现在是 torch tensor，可参与反向传播

#高频
def get_Shigh(synthetic_data, args):
    # 获取节点特征和边索引
    node_features = synthetic_data.x  # [N, d] 节点特征矩阵
    edge_index = synthetic_data.edge_index
    num_nodes = node_features.shape[0]
    feature_dim = node_features.shape[1]
    
    # 计算拉普拉斯矩阵 L = D - A (组合拉普拉斯矩阵)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, 
        num_nodes=num_nodes, 
        normalization=None  # 使用组合拉普拉斯矩阵 L = D - A
    )
    
    # 转换为稠密矩阵
    L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
    # 设置设备
    if args.use_gpu:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    L = L.to(device)
    
    # 用特征矩阵计算S_high_x
    # 随机选取每个节点相同的10%的特征
    num_selected_features = max(1, int(args.feature_prop * feature_dim))  # 至少选择1个特征
    print(f"选取的特征数量{num_selected_features}")
    
    # 随机选择特征索引
    selected_feature_indices = torch.randperm(feature_dim)[:num_selected_features]
    
    S_high_x_i = []
    
    # 对每个选择的特征维度计算 S_high_x_i
    for feature_idx in selected_feature_indices:
        # 选择特征矩阵的第 feature_idx 列作为向量 x
        x = node_features[:, feature_idx]  # [N,] 选择的特征向量
        
        # 计算 S_high_x_i = x^T L x / x^T x
        xTLx = torch.matmul(torch.matmul(x.unsqueeze(0), L), x.unsqueeze(1)).squeeze()  # x^T L x
        xTx = torch.dot(x, x)  # x^T x
        
        # 避免除零
        if xTx > 1e-8:  # 使用小的阈值而不是严格等于0
            S_high_i = xTLx / xTx
            S_high_x_i.append(S_high_i)
    
    # 对所有选取特征的S_high_x_i求平均得到S_high_x
    if len(S_high_x_i) > 0:
        S_high_x = torch.stack(S_high_x_i).mean()
    else:
        S_high_x = torch.tensor(0.0, device=node_features.device)
    

    # 用度矩阵计算 S_high_D 
    # 计算度矩阵D
    # 计算每个节点的度
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    
    # 使用稀疏矩阵操作，不转换为稠密矩阵
    # 构建稀疏拉普拉斯矩阵
    L_indices = edge_index_laplacian
    L_values = edge_weight_laplacian
    L_sparse = torch.sparse_coo_tensor(L_indices, L_values, (num_nodes, num_nodes)).to(synthetic_data.x.device)
    
    S_high_D_i = []
    
    # 对度矩阵的每一列 Di 进行计算，Di 是只有第i个位置为度值deg[i]的向量
    for node_idx in range(num_nodes):
        # 创建度向量 Di: 只有第 node_idx 位置为 deg[node_idx]，其他位置为0
        Di = torch.zeros(num_nodes, device=synthetic_data.x.device)
        Di[node_idx] = deg[node_idx]
        
        # 计算 S_high_D_i = Di^T L Di / Di^T Di
        # 使用稀疏矩阵乘法: L_Di = L * Di
        L_Di = torch.sparse.mm(L_sparse, Di.unsqueeze(1)).squeeze()  # [N,]
        
        # 计算 Di^T L Di = Di^T * L_Di
        DiTLDi = torch.dot(Di, L_Di)  # Di^T L Di
        DiTDi = torch.dot(Di, Di)     # Di^T Di = deg[node_idx]^2
        
        # 避免除零
        if DiTDi > 1e-8:  # 使用小的阈值而不是严格等于0
            S_high_D_i_val = DiTLDi / DiTDi
            S_high_D_i.append(S_high_D_i_val)
    
    # 对所有 S_high_D_i 求平均值得到 S_high_D
    if len(S_high_D_i) > 0:
        S_high_D = torch.stack(S_high_D_i).mean()
    else:
        S_high_D = torch.tensor(0.0, device=node_features.device)
    
    # 计算最终的 S_high (结合特征矩阵和度矩阵的高频分量)
    # 可以采用加权平均或简单平均的方式
    S_high_x_weight = args.S_high_x_weight
    S_high_D_weight = args.S_high_D_weight
    
    print(f"S_high_x: {S_high_x}")
    print(f"S_high_D: {S_high_D}")

    S_high = S_high_x_weight * S_high_x + S_high_D_weight * S_high_D  # 简单平均
    
    return S_high


#计算一个图的同配度(所有边中连接相同类别节点的边的比例)
def get_Homophily_Level(graph):
    # 获取边索引和节点标签
    edge_index = graph.edge_index  # [2, num_edges]
    node_labels = graph.y  # [num_nodes]
    
    # 检查是否有边和标签
    if edge_index.size(1) == 0:
        return 0.0  # 没有边的情况
    
    if node_labels is None:
        return 0.0  # 没有标签的情况
    
    # 获取边的起始和终止节点
    src_nodes = edge_index[0]  # 源节点
    dst_nodes = edge_index[1]  # 目标节点
    
    # 获取边连接的节点的标签
    src_labels = node_labels[src_nodes]  # 源节点的标签
    dst_labels = node_labels[dst_nodes]  # 目标节点的标签
    
    # 计算连接相同类别节点的边数量
    same_label_edges = torch.sum(src_labels == dst_labels).float()
    
    # 计算总边数
    total_edges = edge_index.size(1)
    
    # 计算同配度比例
    homophily_ratio = same_label_edges / total_edges
    
    return homophily_ratio.item()  # 返回标量值



def reduce_homophilic_edges(data, edge_index, reduction_ratio=0.1):
    """减少同配边（相同类别节点之间的边）"""
    labels = data.y
    
    # 找出所有同配边
    homophilic_edges = []
    heterophilic_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] == labels[dst]:  # 同配边
            homophilic_edges.append(i)
        else:  # 异配边
            heterophilic_edges.append(i)
    
    # 随机移除一部分同配边
    num_to_remove = int(len(homophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(homophilic_edges))[:num_to_remove]
        edges_to_remove = [homophilic_edges[i] for i in remove_indices]
        
        # 保留未被移除的边
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index
        
    return new_edge_index

def reduce_heterophilic_edges(data, edge_index, reduction_ratio=0.1):
    """减少异配边（不同类别节点之间的边）"""
    labels = data.y
    
    # 找出所有异配边
    heterophilic_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] != labels[dst]:  # 异配边
            heterophilic_edges.append(i)
    
    # 随机移除一部分异配边
    num_to_remove = int(len(heterophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(heterophilic_edges))[:num_to_remove]
        edges_to_remove = [heterophilic_edges[i] for i in remove_indices]
        
        # 保留未被移除的边
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index
        
    return new_edge_index
