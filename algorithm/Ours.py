from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse, coalesce, get_laplacian
from models.model import *
from datasets.dataset_loader import class_to_task
from utils import Generator, LinkPredictor, edge_distribution_low, construct_graph, DeepInversionHook, compute_led, KDE, gaussian_kernel, apply_kde_to_energy_dist, compute_js_divergence_from_prob_dist, get_SC, get_Shigh, reduce_homophilic_edges, reduce_heterophilic_edges, get_Homophily_Level
from datasets.partition import get_subgraph_by_node
import copy
import matplotlib.pyplot as plt
import os
from datetime import datetime

class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool)
        self.args = args
        self.clients_num = self.args.clients_num
        # self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.device = device
        self.global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = [load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device) for _ in range(self.clients_num)]
        self.last_global_model = load_model(name = args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.per_task_class_num = self.args.per_task_class_num
        # 初始化生成器和链接预测器
        self.noise_dim = getattr(args, 'noise_dim', 128)  # 噪声维度
        #初始化存储客户端信息的列表
        self.clients_nodes_num = []
        self.clients_learned_nodes_num = []
        self.clients_graph_energy = []
        self.clients_mean_var = []
        # 初始化损失值存储列表 - 改为每个任务存储连续的epoch损失
        self.generator_losses = {}  # 存储每个任务所有通信轮次的生成器训练损失值 {task_id: [losses]}
        self.generator_loss_details = {}  # 存储每个任务所有通信轮次的生成器训练详细损失分项 {task_id: [details]}
        self.kd_losses = {}  # 存储每个任务所有通信轮次的知识蒸馏损失值 {task_id: [losses]}
        self.kd_loss_details = {}  # 存储每个任务所有通信轮次的知识蒸馏详细损失分项 {task_id: [details]}

    def aggregate(self):
        """
        聚合客户端模型参数到全局模型
        从message_pool中获取客户端模型参数
        """
        # 从message_pool中获取客户端模型参数
        client_weights = []
        for client_idx in range(self.clients_num):
            client_key = f"client_{client_idx}"
            if client_key in self.message_pool and "weight" in self.message_pool[client_key]:
                client_weights.append(self.message_pool[client_key]["weight"])
        
        # 如果没有获取到客户端参数，则直接返回
        if not client_weights:
            print("No client weights found in message_pool")
            return
            
        totoal_nodes_num = sum([self.message_pool[f"client_{client_id}"]["nodes_num"] for client_id in range(self.clients_num)])

        #更新服务器模型参数
        for i, client_id in enumerate(range(self.clients_num)):
            weight = self.message_pool[f"client_{client_id}"]["nodes_num"] / totoal_nodes_num
            for (client_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.global_model.parameters()):
                if i == 0:
                    global_param.data.copy_(weight * client_param)
                else:
                    global_param.data += weight * client_param


    def feature_gen_init(self, task_id):
        # 计算类别数量：可以是总类别数或者当前任务的类别数
        classes_num = task_id * self.per_task_class_num
        self.generator = Generator(noise_dim = self.noise_dim, input_dim = self.args.input_dim, output_dim = classes_num, dropout = self.args.dropout).to(self.device)

    #生成器参数聚合:
    def feature_gen_aggregate(self):
        """
        聚合客户端的生成器参数
        根据客户端已学习的节点数量进行加权聚合
        """
        #从message_pool中收集生成器、节点数量
        client_generators = []
        client_weights = []
        
        # 收集所有客户端的生成器信息
        for client_id in range(self.clients_num):
            client_gen_key = f"client_{client_id}_generator"
            if client_gen_key in self.message_pool:
                client_gen_info = self.message_pool[client_gen_key]
                
                # 只有当生成器参数存在且已学习节点数大于0时才参与聚合
                if (client_gen_info["generator_weight"] is not None and 
                    client_gen_info["learned_nodes_num"] > 0):
                    client_generators.append(client_gen_info["generator_weight"])
                    client_weights.append(client_gen_info["learned_nodes_num"])
        
        
        #根据节点数量加权更新生成器的参数
        total_nodes = sum(client_weights)
        
        # 归一化权重
        normalized_weights = [weight / total_nodes for weight in client_weights]
        
        # 聚合生成器参数
        with torch.no_grad():
            for param_idx, server_param in enumerate(self.generator.parameters()):
                # 初始化聚合参数
                aggregated_param = torch.zeros_like(server_param)
                
                # 加权求和
                for client_idx, client_gen_params in enumerate(client_generators):
                    client_param = client_gen_params[param_idx]
                    weight = normalized_weights[client_idx]
                    aggregated_param += weight * client_param
                
                # 更新服务器生成器参数
                server_param.data.copy_(aggregated_param)
        
        print(f"生成器参数聚合完成，参与聚合的客户端数量: {len(client_generators)}")
        print(f"客户端权重: {normalized_weights}")

    def synthesis_data(self, task_id, num_samples_per_class=10):
        """
        使用训练好的生成器生成合成数据，根据全局S_high值调整图结构
        
        参数:
        task_id: 当前任务ID
        num_samples_per_class: 每个类别生成的样本数量
        """
        # 1. 从message_pool中收集节点数量和S_high信息
        client_s_highs = []
        client_nodes_nums = []
        
        print("从message_pool收集客户端S_high信息...")
        for client_id in range(self.clients_num):
            client_gen_key = f"client_{client_id}_generator"
            if client_gen_key in self.message_pool:
                client_info = self.message_pool[client_gen_key]
                if client_info["learned_nodes_num"] > 0:  # 只考虑有已学习节点的客户端
                    client_s_highs.append(client_info["S_high"])
                    client_nodes_nums.append(client_info["learned_nodes_num"])
        
        # 2. 根据节点数量加权计算全局S_high
        total_nodes = sum(client_nodes_nums)
        weighted_s_high = sum(s_high * nodes_num for s_high, nodes_num in zip(client_s_highs, client_nodes_nums))
        global_s_high = weighted_s_high / total_nodes if total_nodes > 0 else 0.5
    
        print(f"全局加权S_high: {global_s_high}")
        
        # 3. 生成节点特征
        self.generator.eval()
        classes_num = task_id * self.per_task_class_num  # 截止到当前任务的类别数量
        
        if classes_num == 0:  # 如果是第一个任务，至少生成第一个任务的类别
            classes_num = self.per_task_class_num
            
        with torch.no_grad():
            all_features = []
            all_labels = []
            
            # 为每个已学习的类别生成样本
            for class_id in range(classes_num):
                # 生成噪声
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                # 生成当前类别的标签
                labels = torch.full((num_samples_per_class,), class_id, dtype=torch.long).to(self.device)
                
                # 生成节点特征
                generated_logits = self.generator(noise, labels)
                features = F.normalize(generated_logits, p=2, dim=1)
                all_features.append(features)
                all_labels.append(labels)
            
            # 合并所有特征和标签
            synthetic_features = torch.cat(all_features, dim=0)
            synthetic_labels = torch.cat(all_labels, dim=0)
            
        # 4. 构建初始图结构
        num_nodes = synthetic_features.shape[0]
        num_edges = int(num_nodes * self.args.gen_num_nodes)  # 可以根据需要调整边的数量

        # 随机生成边的起点和终点
        row = torch.randint(0, num_nodes, (num_edges,), device=synthetic_features.device)
        col = torch.randint(0, num_nodes, (num_edges,), device=synthetic_features.device)
        edge_index = torch.stack([row, col], dim=0)


        # 创建初始合成数据
        synthetic_data = Data(x=synthetic_features, edge_index=edge_index, y=synthetic_labels).to(self.device)

        # 5. 计算生成图的S_high值
        current_s_high = get_Shigh(synthetic_data, self.args)
        print(f"初始生成图S_high: {current_s_high}, 目标全局S_high: {global_s_high}")
        
        # 6. 根据S_high比较调整图结构
        tolerance = self.args.tolerance             # 容忍度
        max_iterations = self.args.max_iterations   # 最大调整次数
        
        for iteration in range(max_iterations):
            if abs(current_s_high - global_s_high) <= tolerance:
                print(f"S_high调整完成，迭代{iteration}次")
                break
                
            if current_s_high > global_s_high:
                # S_high大于全局值，减少同配边(相同类之间的节点)
                edge_index = reduce_homophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
            else:
                # S_high小于全局值，减少异配边(不同类之间的节点)
                edge_index = reduce_heterophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
            
            # 更新图数据并重新计算S_high
            synthetic_data.edge_index = edge_index
            current_s_high = get_Shigh(synthetic_data, self.args)
            print(f"调整后第{iteration+1}次S_high: {current_s_high}")
        
        # 7. 创建最终的合成数据任务格式
        num_nodes = synthetic_features.shape[0]
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        
        print("KD Train SYS Data")
        print(f"synthetic_data.x: {synthetic_data.x}")
        print(f"synthetic_data.y: {synthetic_data.y}")
        print(f"synthetic_data.edge_index: {synthetic_data.edge_index}")


        # 构造与客户端任务格式相同的数据结构
        self.synthesis_task = {
            "local_data": synthetic_data,
            "train_mask": train_mask,
            "valid_mask": torch.zeros(num_nodes, dtype=torch.bool),  # 空的验证掩码
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool)     # 空的测试掩码
        }
        
        print(f"合成数据生成完成: {num_nodes}个节点, {edge_index.shape[1]}条边")
        print(f"最终S_high: {current_s_high}, 目标S_high: {global_s_high}")
        


    #上一轮的全局模型对这一轮的全局模型的知识蒸馏
    def KD_train(self, task_id):
        """
        使用生成的合成数据和上一轮的全局模型对当前全局模型进行知识蒸馏
        参数:
        task_id: 当前任务ID
        """
        # 如果是第一个任务，没有需要蒸馏的知识
        if task_id == 0:
            print("First task, no knowledge distillation needed.")
            # 为第一个任务初始化空的KD损失列表
            if task_id not in self.kd_losses:
                self.kd_losses[task_id] = []
                self.kd_loss_details[task_id] = []
            return
            
        # 生成合成数据用于知识蒸馏
        self.synthesis_data(task_id, num_samples_per_class=self.args.num_samples_per_class)

        synthetic_task = self.synthesis_task
        synthetic_data = synthetic_task["local_data"]
        print(f"KD train SYS Data.y:{synthetic_data.y}")

        
        # 设置模型模式
        self.global_model.train()  # 当前全局模型设为训练模式
        self.last_global_model.eval()  # 上一轮全局模型设为评估模式
        
        # 优化器
        kd_optimizer = torch.optim.Adam(self.global_model.parameters(), 
                                       lr=getattr(self.args, 'kd_lr', 0.001))
        
        # 知识蒸馏参数
        temperature = getattr(self.args, 'kd_temperature', 4.0)  # 蒸馏温度
        num_epochs = getattr(self.args, 'kd_epochs', 50)  # 蒸馏轮数
        
        # 将数据移到设备上
        synthetic_data = synthetic_data.to(self.device)
        
        print(f"Starting knowledge distillation for task {task_id}...")
        
        # 为当前任务初始化KD损失记录列表（如果还没有）
        if task_id not in self.kd_losses:
            self.kd_losses[task_id] = []
            self.kd_loss_details[task_id] = []
        
        print(f"========== 全局模型 在第{task_id}个任务训练时的 KD loss值为:==========\n")
        for epoch in range(num_epochs):
            kd_optimizer.zero_grad()
            
            # 当前全局模型的输出（学生模型）
            _, student_logits = self.global_model(synthetic_data)
            
            # 上一轮全局模型的输出（教师模型）
            with torch.no_grad():
                _, teacher_logits = self.last_global_model(synthetic_data)
            
            #计算global_mode在生成数据集上的交叉熵损失
            loss_ce = nn.CrossEntropyLoss()(student_logits, synthetic_data.y)

            #计算低频蒸馏损失
            # 使用温度参数软化输出分布进行知识蒸馏
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
            
            # print(f"KD student_log_probs.requires_grad: {student_log_probs.requires_grad}")
            # print(f"KD teacher_probs.requires_grad: {teacher_probs.requires_grad}")
            
            print("debug nan student_log_probs:", student_log_probs.min(), student_log_probs.max(), torch.isnan(student_log_probs).any(), torch.isinf(student_log_probs).any())
            print("debug nan teacher_probs:", teacher_probs.min(), teacher_probs.max(), torch.isnan(teacher_probs).any(), torch.isinf(teacher_probs).any())

            # 计算低频蒸馏损失（使用edge_distribution_low函数）
            loss_kd_low = edge_distribution_low(
                synthetic_data.edge_index,
                student_log_probs,
                teacher_probs
            )
            
            # 损失权重
            ce_weight = getattr(self.args, 'kd_ce_weight', 0.5)  # 交叉熵损失权重
            low_weight = getattr(self.args, 'kd_low_weight', 0.2) # 低频蒸馏损失权重
            
            # 总损失
            # total_loss = ce_weight * loss_ce
            total_loss = ce_weight * loss_ce + low_weight * loss_kd_low

            # print("KD loss_ce.requires_grad:", loss_ce.requires_grad)
            # print("KD loss_kd_low.requires_grad:", loss_kd_low.requires_grad)
            print("KD loss_kd_low:", loss_kd_low.item())



            print(f"第{epoch}轮的loss为:{total_loss}\n")
            
            # 保存当前epoch的损失值到任务的连续epoch列表中
            current_epoch_in_task = len(self.kd_losses[task_id])
            self.kd_losses[task_id].append(total_loss.item())
            self.kd_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'low_freq_loss': loss_kd_low.item()
            })
            
            # 反向传播
            total_loss.backward()
            kd_optimizer.step()
            
            # 打印训练信息
            if epoch % 10 == 0:
                print(f"KD Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                      f"CE Loss: {loss_ce.item():.4f}, Low Freq Loss: {loss_kd_low.item():.8f}")
        
       

        print("Knowledge distillation completed!")

    def update_last_global_model(self):
        #将last_global_model的参数更新为global_model的参数    
        with torch.no_grad():
            for last_param, global_param in zip(self.last_global_model.parameters(), self.global_model.parameters()):
                last_param.data.copy_(global_param.data)
   
    def plot_loss_curves(self, save_dir="./loss_plots"):
        """
        绘制损失变化图像，包括生成器和知识蒸馏的各项损失
        """
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 绘制生成器总损失变化（每个任务一个图，所有通信轮次连续显示）
        if self.generator_losses:
            for task_id, task_losses in self.generator_losses.items():
                if task_losses:  # 确保任务有损失数据
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Generator Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'generator_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        # 2. 绘制生成器各项损失分解图（每个任务一个图，只包含实际使用的损失）
        if self.generator_loss_details:
            for task_id, task_details in self.generator_loss_details.items():
                if not task_details:  # 跳过空的任务
                    continue
                    
                plt.figure(figsize=(15, 10))
                
                # 提取各项损失（只保留实际使用的）
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                kl_losses = [detail['kl_loss'] for detail in task_details]
                shigh_losses = [detail['shigh_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                # 创建子图
                plt.subplot(2, 3, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 2)
                plt.plot(epochs, kl_losses, 'g-', marker='o', markersize=2)
                plt.title('KL Divergence Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 3)
                plt.plot(epochs, shigh_losses, 'orange', marker='o', markersize=2)
                plt.title('Shigh Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 4)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 5)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, kl_losses, 'g-', label='KL Loss', linewidth=2)
                plt.plot(epochs, shigh_losses, 'orange', label='Shigh Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All Generator Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Generator Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'generator_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. 绘制知识蒸馏总损失变化（每个任务一个图）
        if self.kd_losses:
            for task_id, task_losses in self.kd_losses.items():
                if task_losses:  # 确保任务有损失数据
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Knowledge Distillation Total Loss - Task {task_id + 1} (All Communication Rounds)')
                    plt.xlabel('Continuous Epoch (All Rounds)')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'kd_total_loss_task_{task_id + 1}_{timestamp}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
        # 4. 绘制知识蒸馏各项损失分解图
        if self.kd_loss_details:
            for task_id, task_details in self.kd_loss_details.items():
                if not task_details:  # 跳过空的任务
                    continue
                    
                plt.figure(figsize=(12, 8))
                
                # 提取各项损失
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                low_freq_losses = [detail['low_freq_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                # 创建子图
                plt.subplot(2, 2, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(epochs, low_freq_losses, 'g-', marker='o', markersize=2)
                plt.title('Low Frequency Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, low_freq_losses, 'g-', label='Low Freq Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All KD Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Knowledge Distillation Loss Details - Task {task_id + 1} (All Communication Rounds)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'kd_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Server loss plots saved to: {save_dir}")

    def send_message(self):
        self.message_pool["server"] = {
            "weight" : list(self.global_model.parameters())   
        }
    
    def send_feature_gen(self):
        generator_weight = list(self.generator.parameters())
        
        self.message_pool["server_generator"] = {
            "weight": generator_weight
        }
        print("服务器发送生成器参数到客户端")


class OursClient(BaseClient):   
    def __init__(self, args, client_id, data, message_pool, device):   #data是分割完的子图
        super(OursClient, self).__init__(args, client_id, data)
        self.args = args
        # self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.device = device
        self.global_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.client_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.last_global_model = load_model(name = self.args.model, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, num_layers = args.num_layers, dropout = args.dropout).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()            #交叉熵损失函数 
        self.message_pool = message_pool                #存储一些来自服务器的信息（服务器上模型的参数和服务器生成的数据）
        
        # 保存原始数据用于LED计算
        self.data = data
        
        # 从args中获取任务相关参数
        self.per_task_class_num = getattr(args, 'per_task_class_num', 2)
        self.train_prop = getattr(args, 'train_prop', 0.6)
        self.valid_prop = getattr(args, 'valid_prop', 0.2) 
        self.test_prop = getattr(args, 'test_prop', 0.2)
        self.shuffle_flag = getattr(args, 'shuffle_flag', False)
        
        self.tasks = class_to_task(data = data, per_task_class_num = self.per_task_class_num, train_prop = self.train_prop, valid_prop = self.valid_prop, test_prop = self.test_prop, shuffle_flag = self.shuffle_flag)
        
        print(f"!!!!!!!!!!!!!客户端{client_id}的任务集为：")
        print(f"self.data.x: {self.data.x}")
        print(f"self.data.edge_index: {self.data.edge_index}")
        print(f"self.data.y: {self.data.y}")
        for i, task in enumerate(self.tasks):
            print(f"客户端{self.client_id} - 任务{i}:")
            print(f"  节点总数: {task['local_data'].x.shape[0]}")
            print(f"  边: {task['local_data'].edge_index}")
            print(f"  标签分布: {torch.unique(task['local_data'].y)}")
            print(f"  训练集节点数: {task['train_mask'].sum().item()}")
            print(f"  验证集节点数: {task['valid_mask'].sum().item()}")
            print(f"  测试集节点数: {task['test_mask'].sum().item()}")

            print("  训练集类别:", torch.unique(task['local_data'].y[task['train_mask']]))
            print("  验证集类别:", torch.unique(task['local_data'].y[task['valid_mask']]))
            print("  测试集类别:", torch.unique(task['local_data'].y[task['test_mask']]))
        self.local_epochs = args.local_epochs
        
        # 初始化损失值存储列表 - 改为每个任务存储连续的epoch损失
        self.client_losses = {}  # 存储每个任务所有通信轮次的客户端训练损失值 {task_id: [losses]}
        self.generator_losses = {}  # 存储每个任务所有通信轮次的生成器训练损失值 {task_id: [losses]}
        self.generator_loss_details = {}  # 存储每个任务所有通信轮次的生成器训练详细损失分项 {task_id: [details]}

        self.noise_dim = getattr(args, 'noise_dim', 128)  # 噪声维度

    #本地模型训练，只使用本地数据
    def train(self, task_id):
        # 使用服务器发送的全局模型参数来更新本地模型 self.client_model 和 self.global_model 的参数。
        # 只有在服务器已经发送参数时才更新
        if "server" in self.message_pool and "weight" in self.message_pool["server"]:
            with torch.no_grad():       #客户端的局部模型
                for (local_param_old, agg_global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
            with torch.no_grad():       #客户端的全局模型
                for (local_param_old, agg_global_param) in zip(self.global_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param_old.data.copy_(agg_global_param)
        
        task = self.tasks[task_id]
        global_model = self.global_model                        #使用已加载参数的全局模型

        self.client_model.train()           # 设置模型为训练模式
        global_model.eval()                 # 设置全局模型为评估模式
        
        # 获取本地任务数据
        local_data = task["local_data"]
        local_train_mask = task["train_mask"]
        whole_data = self.data.to(self.device)
        # 将数据移到设备上
        local_data = local_data.to(self.device)
        
        # 配置优化器
        optimizer = torch.optim.Adam(self.client_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        # 训练模型
        print(f"=========={self.client_id} 在第{task_id}个任务训练时的loss值为:==========\n")
        
        # 为当前任务初始化损失记录列表（如果还没有）
        if task_id not in self.client_losses:
            self.client_losses[task_id] = []
        
        for epoch in range(self.local_epochs):
            # 清除梯度
            optimizer.zero_grad()
            
            # 1. 在本地数据上的训练
            _, local_student_out = self.client_model(local_data)
            #交叉熵损失
            loss = self.loss_fn(local_student_out[local_train_mask], whole_data.y[local_train_mask])
            # print("GCN loss.requires_grad:", loss.requires_grad)  true
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 保存当前epoch的损失值到任务的连续epoch列表中
            self.client_losses[task_id].append(loss.item())
            
            print(f"第{epoch}轮的loss为:{loss}\n")

        return loss.item()
    
    def feature_gen_init(self, task_id):
        # 计算类别数量：可以是总类别数或者当前任务的类别数
        classes_num = task_id * self.per_task_class_num
        self.generator = Generator(noise_dim = self.noise_dim, input_dim = self.args.input_dim, output_dim = classes_num, dropout = self.args.dropout).to(self.device)

    def get_learned_graph(self, task_id):
        # 需要从0开始到task_id - 1的所有task中包含的节点
        # 如果是第一个任务，没有之前学习的任务，跳过生成器训练
        if task_id == 0:
            print(f"任务{task_id}是第一个任务，没有之前学习的节点可用于生成器训练")
            return
        
        # 获取从任务0到任务task_id-1的所有已学习节点
        all_nodes = []
        learned_classes = []
        
        # 收集从任务0到任务task_id-1的所有节点
        for prev_task_id in range(task_id):
            task_data = self.tasks[prev_task_id]
            # 获取该任务中所有节点（训练+验证+测试）
            task_nodes_mask = task_data["train_mask"] | task_data["valid_mask"] | task_data["test_mask"]
            task_nodes_indices = torch.where(task_nodes_mask)[0].tolist()
            all_nodes.extend(task_nodes_indices)
            
            # 记录该任务包含的类别
            task_classes = torch.unique(self.data.y[task_nodes_indices]).tolist()
            learned_classes.extend(task_classes)
        
        # 去重并排序
        all_nodes = sorted(list(set(all_nodes)))
        learned_classes = sorted(list(set(learned_classes)))
        
        # 获取子图
        subgraph = get_subgraph_by_node(self.data, all_nodes, True)

        return subgraph

    #计算已学习的任务中train_mask节点的同配度
    def get_learned_Homophily_Level(self, task_id):
        """
        计算已学习任务中训练节点的同配度
        
        参数:
        task_id: 当前任务ID
        
        返回:
        homophily_ratio: 同配度比例，范围 [0, 1]
        """
        total_edge_nums = 0
        same_label_edges = 0
        
        # 遍历从任务0到task_id-1的所有已学习任务
        for task_i in range(task_id):
            task = self.tasks[task_i]  # 修正：应该是task_i而不是task_id
            graph = task["local_data"]
            train_mask = task["train_mask"]
            
            # 获取边索引和节点标签
            edge_index = graph.edge_index  # [2, num_edges]
            node_labels = graph.y  # [num_nodes]
            
            # 遍历graph的所有边，如果一条边的两个节点都在task["train_mask"]中则 total_edge_nums ++
            for edge_idx in range(edge_index.size(1)):
                src_node = edge_index[0, edge_idx].item()
                dst_node = edge_index[1, edge_idx].item()
                
                # 检查两个节点是否都在训练集中
                if train_mask[src_node] and train_mask[dst_node]:
                    total_edge_nums += 1
                    
                    # 如果一条边的两个节点都在task["train_mask"]中且两个节点的标签相同则 same_label_edges ++
                    if node_labels[src_node] == node_labels[dst_node]:
                        same_label_edges += 1
        
        # 计算同配度比例
        if total_edge_nums > 0:
            homophily_ratio = same_label_edges / total_edge_nums
        else:
            homophily_ratio = 0.0  # 没有有效边的情况
            
        return homophily_ratio

    
    def feature_gen_train(self, task_id, global_flag = True):
        """
        特征生成器训练函数
        参数:
        task_id: 当前任务ID
        global_flag: 是否使用服务器生成器的参数更新本地生成器
        """
        # 如果 global_flag = True 则使用服务器生成器的参数更新本地生成器
        if global_flag and "server_generator" in self.message_pool:
            with torch.no_grad():
                for local_param, server_param in zip(self.generator.parameters(), self.message_pool["server_generator"]["weight"]):
                    local_param.data.copy_(server_param)

        subgraph = self.get_learned_graph(task_id)
        subgraph_homo = self.get_learned_Homophily_Level(task_id)
        
        
        # 如果没有已学习的子图，跳过生成器训练
        if subgraph is None:
            print(f"客户端{self.client_id}在任务{task_id}上没有找到已学习的子图，跳过生成器训练")
            return
            
        # 获取已学习的类别
        learned_classes = torch.unique(subgraph.y).tolist()
        
        # 训练生成器
        self.generator.train()
        self.last_global_model.eval()
        
        # 配置优化器和训练参数
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                       lr=getattr(self.args, 'gen_lr', 0.001))
        
        # 获取训练参数
        gen_epochs = getattr(self.args, 'gen_epochs', 50)
        num_samples_per_class = getattr(self.args, 'gen_num_samples_per_class', 200)
        
        # 检查是否有已学习的类别
        if not learned_classes:
            print(f"客户端{self.client_id}在任务{task_id}上没有找到已学习的类别，跳过生成器训练")
            return
            
        # 移动数据到设备
        subgraph = subgraph.to(self.device)
        
        print(f"客户端{self.client_id}在任务{task_id}上开始生成器训练...")
        print(f"已学习的类别: {learned_classes}")
        
        # 为当前任务初始化生成器损失记录列表（如果还没有）
        if task_id not in self.generator_losses:
            self.generator_losses[task_id] = []
            self.generator_loss_details[task_id] = []
        
        for epoch in range(gen_epochs):
            gen_optimizer.zero_grad()
            
            # 进行随机特征生成
            generated_features_list = []
            generated_labels_list = []
            
            for class_idx in learned_classes:
                # 为每个已学习的类别生成样本
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                class_labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(self.device)
                
                # 生成特征
                generated_logits = self.generator(noise, class_labels)
                generated_features = F.normalize(generated_logits, p=2, dim=1)

                generated_features_list.append(generated_features)
                generated_labels_list.append(class_labels)
            
            # 合并所有生成的特征和标签
            all_generated_features = torch.cat(generated_features_list, dim=0)
            all_generated_labels = torch.cat(generated_labels_list, dim=0)
            print("Client Gen Train")
            print("all_generated_features:", all_generated_features)
            print("all_generated_labels:", all_generated_labels)
            
            # 构建生成数据的图结构
            num_nodes = all_generated_features.shape[0]
            num_edges = int(num_nodes * self.args.gen_num_nodes)  # 可以根据需要调整边的数量

            # 随机生成边的起点和终点
            row = torch.randint(0, num_nodes, (num_edges,), device=all_generated_features.device)
            col = torch.randint(0, num_nodes, (num_edges,), device=all_generated_features.device)
            edge_index = torch.stack([row, col], dim=0)


            # 创建初始合成数据
            synthetic_data = Data(x=all_generated_features, edge_index=edge_index, y=all_generated_labels).to(self.device)

            # 5. 计算生成图的 同配度 值
            current_homo = get_Homophily_Level(synthetic_data)
            print(f"初始生成图homo: {current_homo}, 目标homo: {subgraph_homo}")
            
            # 6. 根据S_high比较调整图结构
            tolerance = self.args.tolerance             # 容忍度
            max_iterations = self.args.max_iterations   # 最大调整次数
            
            for iteration in range(max_iterations):
                if abs(current_homo - subgraph_homo) <= tolerance:
                    print(f"S_high调整完成，迭代{iteration}次")
                    break
                    
                if current_homo > subgraph_homo:
                    edge_index = reduce_homophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
                else:
                    edge_index = reduce_heterophilic_edges(synthetic_data, edge_index, reduction_ratio=self.args.gen_reduction_ratio)
                
                synthetic_data.edge_index = edge_index
                current_homo = get_Homophily_Level(synthetic_data)
                print(f"第{iteration}调整后的Homo: {current_homo}")

            # 使用全局模型进行预测，然后进行CE_loss计算
            # with torch.no_grad():
            #     _, global_predictions = self.last_global_model(synthetic_data)
            _, global_predictions = self.last_global_model(synthetic_data)
            
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss_ce = cross_entropy_loss(global_predictions, all_generated_labels)
            
            # 计算每个生成节点与真实数据中随机一个同类节点的特征之间的KL散度
            kl_losses = []
            for i, gen_label in enumerate(all_generated_labels):
                # 找到真实数据中相同类别的节点
                same_class_mask = (subgraph.y == gen_label)
                if same_class_mask.sum() > 0:
                    # 随机选择一个同类节点
                    same_class_indices = torch.where(same_class_mask)[0]
                    random_idx = torch.randint(0, len(same_class_indices), (1,)).item()
                    real_feature = subgraph.x[same_class_indices[random_idx]]
                    
                    kl_loss = F.kl_div(
                        F.log_softmax(all_generated_features[i]),
                        F.softmax(real_feature),
                        reduction = 'sum'
                    )

                    kl_losses.append(kl_loss)
            
            if kl_losses:
                loss_kl = torch.stack(kl_losses).mean()
            else:
                loss_kl = torch.tensor(0.0, device=self.device)
            
            # 总损失计算
            ce_weight = getattr(self.args, 'gen_ce_weight', 1.0)
            kl_weight = getattr(self.args, 'gen_kl_weight', 0.5)
            
            # total_loss = ce_weight * loss_ce 
            total_loss = ce_weight * loss_ce + kl_weight * loss_kl 
            
            # print(f"loss_ce.requires_grad: {loss_ce.requires_grad}")
            # print(f"loss_shigh.requires_grad: {loss_shigh.requires_grad}")
            # print(f"loss_kl.requires_grad: {loss_kl.requires_grad}")
            # 记录损失值到任务的连续epoch列表中
            current_epoch_in_task = len(self.generator_losses[task_id])
            self.generator_losses[task_id].append(total_loss.item())
            self.generator_loss_details[task_id].append({
                'epoch': current_epoch_in_task,
                'total_loss': total_loss.item(),
                'ce_loss': loss_ce.item(),
                'kl_loss': loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl,
                'shigh_loss': 0.0
            })
            
            # 反向传播
            total_loss.backward()
            gen_optimizer.step()
            
            # 记录损失 (如果需要的话)
            if epoch % 10 == 0:
                print(f"SYS Epoch {epoch}/{gen_epochs}, "
                      f"总损失: {total_loss.item():.4f}, "
                      f"CE损失: {loss_ce.item():.4f}, "
                      f"KL损失: {loss_kl.item():.4f}")
        
        print(f"客户端{self.client_id}在任务{task_id}上的生成器训练完成!")

    def plot_generator_loss_curves(self, save_dir="./loss_plots"):
        """
        绘制客户端生成器损失变化图像
        """
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 绘制生成器总损失变化（每个任务一个图）
        if self.generator_losses:
            for task_id, task_losses in self.generator_losses.items():
                if task_losses:  # 确保任务有损失数据
                    plt.figure(figsize=(12, 8))
                    epochs = range(len(task_losses))
                    plt.plot(epochs, task_losses, label=f'Task {task_id + 1}', marker='o', markersize=3)
                    
                    plt.title(f'Client {self.client_id} - Generator Total Loss - Task {task_id + 1}')
                    plt.xlabel('Continuous Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'client_{self.client_id}_generator_loss_task_{task_id + 1}_{timestamp}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 2. 绘制生成器各项损失分解图
        if self.generator_loss_details:
            for task_id, task_details in self.generator_loss_details.items():
                if not task_details:  # 跳过空的任务
                    continue
                    
                plt.figure(figsize=(15, 10))
                
                # 提取各项损失
                epochs = [detail['epoch'] for detail in task_details]
                ce_losses = [detail['ce_loss'] for detail in task_details]
                kl_losses = [detail['kl_loss'] for detail in task_details]
                shigh_losses = [detail['shigh_loss'] for detail in task_details]
                total_losses = [detail['total_loss'] for detail in task_details]
                
                # 创建子图
                plt.subplot(2, 3, 1)
                plt.plot(epochs, ce_losses, 'b-', marker='o', markersize=2)
                plt.title('Cross Entropy Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 2)
                plt.plot(epochs, kl_losses, 'g-', marker='o', markersize=2)
                plt.title('KL Divergence Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 3)
                plt.plot(epochs, shigh_losses, 'orange', marker='o', markersize=2)
                plt.title('Shigh Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 4)
                plt.plot(epochs, total_losses, 'k-', marker='o', markersize=2)
                plt.title('Total Loss')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 3, 5)
                plt.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
                plt.plot(epochs, kl_losses, 'g-', label='KL Loss', linewidth=2)
                plt.plot(epochs, shigh_losses, 'orange', label='Shigh Loss', linewidth=2)
                plt.plot(epochs, total_losses, 'k-', label='Total Loss', linewidth=2)
                plt.title('All Generator Losses Combined')
                plt.xlabel('Continuous Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Client {self.client_id} - Generator Loss Details - Task {task_id + 1}', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'client_{self.client_id}_generator_detailed_loss_task_{task_id + 1}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Client {self.client_id} generator loss plots saved to: {save_dir}")

    #获得已经学习过的节点子图
    def get_subgraph(self, task_id):
        """
        计算拉普拉斯能量分布(LED)值
        根据公式: \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
        """
        # 如果是第一个任务，返回0（没有之前学习的节点）
        if task_id == 0:
            return 0.0
            
        #根据任务获得截止到task_id时已学习的节点种类
        nodes_list = []              #节点编号列表
        
        # 需要获取原始完整数据
        original_data = self.data 
        
        for i in range(task_id): #已经学习的任务编号是[0, task_id - 1]
            task = self.tasks[i]
            nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
            
            # 获取当前任务中所有节点的索引
            nodes_indices = torch.where(nodes_mask)[0].tolist()
            nodes_list.extend(nodes_indices)
        
        # 去重并排序，得到所有已学习的节点列表
        # nodes_list = sorted(list(set(nodes_list)))
        
        # 获取包含已学习节点的子图
        subgraph = get_subgraph_by_node(original_data, nodes_list, True)

        return subgraph

    #获得LED(Laplacian Energy Distribution)值
    def get_LED(self, subgraph):
        # 直接计算LED
        nodes_feature = subgraph.x  # [N, d] 节点特征矩阵
        edge_index = subgraph.edge_index
        
        # 1. 计算图的拉普拉斯矩阵
        num_nodes = nodes_feature.shape[0]
        # 获取标准化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
        edge_index_laplacian, edge_weight_laplacian = get_laplacian(
            edge_index, 
            num_nodes=num_nodes, 
            normalization='sym'  # 对称标准化
        )
        
        # 转换为稠密矩阵
        L = to_dense_adj(edge_index_laplacian, edge_attr=edge_weight_laplacian, max_num_nodes=num_nodes)[0]
        
        # 2. 对拉普拉斯矩阵进行特征值分解，获取特征向量矩阵 U
        eigenvalues, eigenvectors = torch.linalg.eigh(L)  # U: [N, N]
        U = eigenvectors  # 特征向量矩阵
        U = U.to(self.device)
        nodes_feature = nodes_feature.to(self.device)
        # 3. 计算图傅里叶变换 \hat{X} = U^T X
        X_hat = torch.matmul(U.T, nodes_feature)  # [N, d] 傅里叶变换后的特征
        
        # 4. 根据公式计算 LED 值
        # \bar{x}_n = \frac{\hat{x}_n^2}{\sum_{i=1}^N \hat{x}_i^2}
        
        # 计算每个频率分量的能量（所有特征维度的平方和）
        energy_per_freq = torch.sum(X_hat ** 2, dim=1)  # [N,] 每个频率的能量
        print(f"led debug_info energy_per_freq: {energy_per_freq}")
        # 计算总能量
        total_energy = torch.sum(energy_per_freq)       #\sum_{i=1}^N \hat{x}_i^2
        
        # 计算能量分布（归一化）
        if total_energy > 0:
            energy_distribution = energy_per_freq / total_energy  # [N,] 归一化的能量分布
        else:
            energy_distribution = torch.zeros_like(energy_per_freq)
        
        return energy_distribution
        
    def evaluate(self, task_id, global_flag = True, mask = "test_mask"):
        """评估客户端模型在指定任务上的性能"""
        task = self.tasks[task_id]
        
        if global_flag:
            print(f"使用全局模型评估{task_id}")
            client_param_copy = copy.deepcopy(list(self.client_model.parameters()))
            with torch.no_grad():
                for(client_param, global_param) in zip(self.client_model.parameters(), self.message_pool["server"]["weight"]):
                    client_param.data.copy_(global_param)

        # 设置模型为评估模式
        self.client_model.eval()
        
        # 获取任务数据
        data = task["local_data"]
        
        # 将数据移到设备上
        data = data.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _, out = self.client_model(data)
            
            # 计算验证集损失
            loss = self.loss_fn(out[task[mask]], data.y[task[mask]])
            
            # 计算验证集精度
            _, pred = out.max(dim=1)
            correct = pred[task[mask]].eq(data.y[task[mask]]).sum().item()
            acc = (correct / task[mask].sum().item()) * 100
        
        # 在 evaluate 函数的前向传播后加：
        print("真实标签:", data.y[task[mask]])
        print("预测标签:", pred[task[mask]])
        print("类别分布:", torch.unique(data.y[task[mask]]), torch.unique(pred[task[mask]]))

        if global_flag:
            with torch.no_grad():
                for(global_param, client_param) in zip(self.client_model.parameters(), client_param_copy):
                    global_param.data.copy_(client_param)
        
        return {"loss": loss.item(), "acc": acc}
    
    def test(self, task_id):
        """测试客户端模型在指定任务的测试集上的性能"""
        task = self.tasks[task_id]
        
        # 设置模型为评估模式
        self.client_model.eval()
        
        # 获取任务数据
        data = task["local_data"]
        test_mask = task["test_mask"]
        
        # 将数据移到设备上
        data = data.to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _, out = self.client_model(data)
            
            # 计算测试集精度
            _, pred = out.max(dim=1)
            correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
            
        return {"acc": acc}
    
    def update_global_model(self):
        """从服务器获取全局模型参数"""
        self.global_model.load_state_dict(self.message_pool["global_model_params"])
    
    def update_client_model(self):
        """将客户端模型更新为全局模型"""
        self.client_model.load_state_dict(self.global_model.state_dict())

    #获取当前任务中客户端数据的节点数量
    def get_task_nodes_num(self, task_id):
        task = self.tasks[task_id]
        nodes_mask = task["train_mask"] | task["valid_mask"] | task["test_mask"]
        nodes_num = nodes_mask.sum()
        return nodes_num 

    def get_mean_var(self, task_id):
        task = self.tasks[task_id]
        local_data = task["local_data"] 
        #计算当前task_id中的data.x的均值和方差
        
        # 获取当前任务的训练节点特征
        # train_mask = task["train_mask"]
        # node_features = local_data.x[train_mask]  # 只使用训练节点的特征
        node_features = local_data.x
        
        # 计算均值和方差
        # 沿着节点维度(dim=0)计算，得到每个特征维度的统计信息
        mean = torch.mean(node_features, dim=0)  # [feature_dim]
        var = torch.var(node_features, dim=0, unbiased=False)  # [feature_dim], 使用总体方差
        
        return {
            "mean": mean,  # 均值
            "var": var     # 方差
        }

    #向服务器发送信息
    def send_message(self, task_id):
        if task_id == 0:
            learned_nodes_num = 0
            data_LED = 0.0
        else:
            subgraph = self.get_subgraph(task_id = task_id)
            learned_nodes_num = subgraph.x.shape[0]
            data_LED = self.get_LED(subgraph)

        self.message_pool[f"client_{self.client_id}"] = {
            "nodes_num" : self.get_task_nodes_num(task_id),           #节点数量
            "learned_nodes_num" : learned_nodes_num,                  #已学习的节点数量 
            "data_LED" : data_LED,                                    #本地数据的
            "mean_var" : self.get_mean_var(task_id),                  #本地数据的均值和方差信息
            "weight" : list(self.client_model.parameters())
        }

    def send_feature_gen(self, task_id):
        # 获取已学习的子图
        subgraph = self.get_learned_graph(task_id)
        
        # 计算S_high值
        S_high = get_Shigh(subgraph, self.args)
        
        # 计算已学习的节点数量
        learned_nodes_num = subgraph.x.shape[0]
        
        # 获取生成器参数（如果生成器已经初始化）
        generator_weight = list(self.generator.parameters())
        
        # 发送到message_pool
        self.message_pool[f"client_{self.client_id}_generator"] = {
            "generator_weight": generator_weight,           # 生成器模型参数
            "learned_nodes_num": learned_nodes_num,         # 已学习的节点数量
            # "S_high": S_high.item() if torch.is_tensor(S_high) else S_high  # 已学习子图的S_high值
            "S_high": S_high  # 已学习子图的S_high值
        }
        
        print(f"客户端{self.client_id}在任务{task_id}上发送生成器信息:")
        print(f"  已学习节点数: {learned_nodes_num}")
        # print(f"  S_high值: {S_high.item() if torch.is_tensor(S_high) else S_high}")
        print(f"  S_high值: {S_high}")
        print(f"  生成器参数: {'已发送' if generator_weight is not None else '未初始化'}")


def plot_all_losses(server, clients, save_dir="./loss_plots"):
    """
    绘制所有损失变化图像，包括服务器和客户端的损失
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 绘制服务器损失
    server.plot_loss_curves(save_dir)
    
    # 1.5. 绘制所有客户端的生成器损失
    for client in clients:
        if hasattr(client, 'generator_losses') and client.generator_losses:
            client.plot_generator_loss_curves(save_dir)
    
    # 2. 绘制所有客户端的本地训练损失
    if clients and clients[0].client_losses:
        # 获取所有任务ID
        all_task_ids = set()
        for client in clients:
            all_task_ids.update(client.client_losses.keys())
        
        # 为每个任务绘制所有客户端的损失对比图
        for task_id in sorted(all_task_ids):
            plt.figure(figsize=(12, 8))
            
            for client in clients:
                if task_id in client.client_losses and client.client_losses[task_id]:
                    epochs = range(len(client.client_losses[task_id]))
                    plt.plot(epochs, client.client_losses[task_id], 
                            label=f'Client {client.client_id}', 
                            marker='o', markersize=3)
            
            plt.title(f'Client Local Training Loss - Task {task_id + 1} (All Communication Rounds)')
            plt.xlabel('Continuous Epoch (All Rounds)')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'clients_local_loss_task_{task_id + 1}_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 绘制每个客户端跨任务的损失变化
        for client in clients:
            if client.client_losses:
                plt.figure(figsize=(12, 8))
                
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        epochs = range(len(task_losses))
                        plt.plot(epochs, task_losses, 
                                label=f'Task {task_id + 1}', 
                                marker='o', markersize=3)
                
                plt.title(f'Client {client.client_id} - Local Training Loss Over Tasks (Each Task All Rounds)')
                plt.xlabel('Continuous Epoch (All Rounds per Task)')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'client_{client.client_id}_loss_over_tasks_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. 绘制所有客户端和所有任务的损失综合图
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(clients)))
        
        for client_id, client in enumerate(clients):
            if client.client_losses:
                for task_id, task_losses in client.client_losses.items():
                    if task_losses:
                        # 为了在全局图中显示，需要调整x轴（考虑任务间的间隔）
                        global_epochs = [epoch + task_id * len(task_losses) * 1.2 for epoch in range(len(task_losses))]
                        plt.plot(global_epochs, task_losses, 
                                color=colors[client_id], 
                                linestyle='-' if task_id == 0 else '--' if task_id == 1 else ':',
                                label=f'Client {client_id} Task {task_id + 1}', 
                                marker='o', markersize=2)
        
        plt.title('All Clients Local Training Loss - All Tasks (Each Task All Rounds)')
        plt.xlabel('Global Training Progress (Continuous Epochs)')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_clients_all_tasks_loss_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All loss plots saved to: {save_dir}")
    return save_dir

#加载服务器和客户端
def load_server_clients(args, data, device):
    message_pool = {}
    clients_num = args.clients_num
    server = OursServer(args, message_pool, device)
    clients = [OursClient(args, client_id, data[client_id], message_pool, device) for client_id in range(clients_num)]

    return server, clients, message_pool