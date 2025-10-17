import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import networkx as nx
from community import community_louvain

# 导入自定义模块
from args import parser
from models.model import load_model
from algorithm.Ours import load_server_clients, plot_all_losses
from datasets.partition import get_subgraph_by_node,louvain_partitioner
from datasets.dataset_loader import load_dataset, class_to_task
from utils import set_seed, AA, AF


def main():
    args = parser.parse_args()
    print(f"换了种子{args.seed} 数据集: {args.dataset_name} 模型: {args.model}")
    print("更新了client gen 中损失函数之间的权重 以及 模仿了Ghost")
    print(f" gen_lr = {args.gen_lr}  kd_lr = {args.kd_lr} SYS epochs = {args.gen_epochs} kd_epochs = {args.kd_epochs}")
    print(f" gen_rounds = {args.gen_rounds}")
    # print("看一下lr_g = 0.001 gen_rounds = 10 时本地生成器多少轮能收敛")
    print("Gen和蒸馏都是两个损失函数都用了")
    print(f"kd_ce_weight: {args.kd_ce_weight}           kd_low_weight: {args.kd_low_weight}      temperature: {args.kd_temperature}")
    print(f"gen_ce_weight: {args.gen_ce_weight}         gen_kl_weight: {args.gen_kl_weight}")
    print(f"S_high_x_weight: {args.S_high_x_weight}     S_high_D_weight: {args.S_high_D_weight}")

    print("参数不变， 只是换了测试集分割比例 GAT修正")
    # 解析参数
    print(f"参数是：{args}")
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.use_gpu:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    
    # 加载数据集
    print(f"Loading dataset: {args.dataset_name}")
    data, input_dim, out_dim, train_valid_test_split= load_dataset(args.dataset_dir, args.dataset_name, args.per_task_class_num, args)
    args.train_prop = train_valid_test_split[0]
    args.valid_prop = train_valid_test_split[1]
    args.test_prop = train_valid_test_split[2]
    print(f"train_prop = {args.train_prop} valid_prop = {args.valid_prop} test_prop = {args.test_prop}")
    
    args.input_dim = input_dim
    args.output_dim = out_dim

    # 打印数据集信息
    print(f"Dataset: {args.dataset_name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.x}")
    print(f"Number of classes: {data.y.max().item() + 1}")
    
   
    
    # 使用社区检测算法划分客户端
    print("Partitioning graph for federated learning...")
   
    clients_data = louvain_partitioner(data, args.clients_num)

    args.input_dim = input_dim
    args.out_dim = out_dim
    server, clients, message_pool = load_server_clients(args, clients_data, device)
    
    # 打印实验设置
    print(f"\nExperiment settings:")
    print(f"Clients: {args.clients_num}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Model: {args.model}")
    
    tasks_num = (data.y.max().item() + 1) // args.per_task_class_num
    
    print(f"Total number of tasks: {tasks_num}")
    
    # 存储每轮训练结果
    ACC_matrix = torch.zeros(size = (tasks_num, tasks_num)).to(device)

    # 对于每个任务
    for task_id in range(tasks_num):
        print(f"\n========== Task {task_id}/{tasks_num} ==========")

        # GNN训练
        for round in range(args.rounds):
            print(f"******************************GNN的第 {round} 轮*************************************")
            # 客户端训练
            print(f"Training clients for task {task_id}...")
            client_losses = []
            for client_id, client in enumerate(clients):
                print(f"  Training client {client_id}/{args.clients_num}...")
                loss = client.train(task_id)
                client_losses.append(loss)
                print(f"Client {client_id} loss: {loss:.4f}")
            
            avg_client_loss = sum(client_losses) / len(client_losses)
            print(f"Average client loss: {avg_client_loss:.4f}")

            # 客户端发送信息
            print("Clients sending information to server...")
            for client_id, client in enumerate(clients):
                client.send_message(task_id)
                print(f"Client {client_id} sent model parameters and LED info")
            
            # 收集客户端信息用于后续处理
            server.clients_nodes_num = [server.message_pool[f"client_{client_id}"]["nodes_num"] 
                                    for client_id in range(args.clients_num)]
            server.clients_learned_nodes_num = [server.message_pool[f"client_{client_id}"]["learned_nodes_num"] 
                                    for client_id in range(args.clients_num)]
            # print(f"Debug Info server.clients_nodes_num:{server.clients_nodes_num}")    正确
            server.clients_graph_energy = [server.message_pool[f"client_{client_id}"]["data_LED"] 
                                        for client_id in range(args.clients_num)]
            # print(f"Debug Info server.clients_graph_energy:{server.clients_graph_energy}") 正确

            # 全局模型更新参数
            print("Server aggregating client models...")
            server.aggregate()
            
            # 全局模型下发模型参数
            print("Server broadcasting updated global model...")
            server.send_message()


        #生成器训练
        # 如果不是第一个任务，进行生成器训练、数据生成和知识蒸馏
        if task_id != 0:
            #初始化生成器
            server.feature_gen_init(task_id = task_id)
            for client_id, client in enumerate(clients):
                client.feature_gen_init(task_id = task_id)
                
            for round in range(args.gen_rounds):
                print(f"******************************生成器的第 {round} 轮*************************************")
                # 客户端训练特征生成器
                for client_id, client in enumerate(clients):
                    client.feature_gen_train(task_id, round != 0)

                    client.send_feature_gen(task_id)
                
                # 服务器聚合特征生成器
                server.feature_gen_aggregate()

                server.send_feature_gen()

            # 生成数据，知识蒸馏
            server.KD_train(task_id)

        # 更新last_global（服务器以及客户端）
        server.update_last_global_model()
        for client_id, client in enumerate(clients):
            client.last_global_model.load_state_dict(server.last_global_model.state_dict())
        

        # 评估模型
        for eval_task_id in range(0, task_id + 1):
            total_nodes_num = 0
            for client_id in range(args.clients_num):
                evaluation = clients[client_id].evaluate(task_id = eval_task_id, global_flag=True)
                client_acc = evaluation["acc"]
                nodes_num = clients[client_id].tasks[eval_task_id]["test_mask"].sum()
                print(f"调试: client_id={client_id}, eval_task_id={eval_task_id}, acc={client_acc}, nodes_num={nodes_num}")
                ACC_matrix[task_id, eval_task_id] += client_acc * nodes_num
                total_nodes_num += nodes_num

            ACC_matrix[task_id, eval_task_id] /= total_nodes_num
            # print(f"******* task_id:{task_id} eval_task_id: {eval_task_id}  ACC_matrix: {ACC_matrix[task_id, eval_task_id]} ********")
            
            print(f"任务{task_id}训练完成后，任务{eval_task_id}的全局准确率为：{ACC_matrix[task_id, eval_task_id]:.2f}\n")

        print(ACC_matrix)

        aa = AA(ACC_matrix, T = task_id + 1)
        af = AF(ACC_matrix, T = task_id + 1)
        print(f"任务{task_id}训练完成后，全局AA为:{aa}\t,全局AF为:{af}")
    
    # 训练完成后绘制所有损失变化图像
    print("\n========== 绘制损失变化图像 ==========")
    loss_plots_dir = plot_all_losses(server, clients, save_dir=args.save_dir)
    print(f"所有损失变化图像已保存到: {loss_plots_dir}")
    
    print("\n========== 训练完成 ==========")
    print(f"最终准确率矩阵:")
    print(ACC_matrix)
    print(f"最终AA: {AA(ACC_matrix, T = tasks_num)}")
    print(f"最终AF: {AF(ACC_matrix, T = tasks_num)}")


if __name__ == "__main__":
    main()