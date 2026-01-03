# import os
# import json
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
#     classification_report, roc_auc_score
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm  # 引入进度条模块
# import random
# import numpy as np
# import logging
#
# from model import swin_small_patch4_window7_224 as create_model
#
#
# def setup_logging(log_file):
#     """设置日志记录到文件"""
#     logging.basicConfig(filename=log_file,
#                         format='%(asctime)s - %(message)s',
#                         level=logging.INFO)
#
#
# def load_image(img_path, data_transform):
#     assert os.path.exists(img_path), "文件 '{}' 不存在.".format(img_path)
#     img = Image.open(img_path).convert('RGB')
#     img = data_transform(img)
#     return img
#
#
# # def evaluate_model(model, dataloader, class_indict, device, log_file=None, save_dir=None):
# #     model.eval()
# #     all_preds = []
# #     all_labels = []
# #     all_scores = []
# #     losses = []
# #
# #     with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:  # 设置进度条总长度
# #         for inputs, labels in dataloader:
# #             inputs = inputs.to(device)
# #             labels = labels.to(device)
# #             outputs = model(inputs)
# #             _, preds = torch.max(outputs, 1)
# #             all_preds.extend(preds.cpu().numpy())
# #             all_labels.extend(labels.cpu().numpy())
# #             all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
# #
# #             loss = torch.nn.functional.cross_entropy(outputs, labels)
# #             losses.append(loss.item())
# #
# #             progress_bar.update(1)  # 更新进度条
# #
# #     accuracy = accuracy_score(all_labels, all_preds)
# #     precision = precision_score(all_labels, all_preds, average='weighted')
# #     recall = recall_score(all_labels, all_preds, average='weighted')
# #     f1 = f1_score(all_labels, all_preds, average='weighted')
# #
# #     # 混淆矩阵
# #     cm = confusion_matrix(all_labels, all_preds)
# #
# #     # 分类报告
# #     cls_report = classification_report(all_labels, all_preds, target_names=list(class_indict.values()),
# #                                        output_dict=True)
# #
# #     # Top-1 准确率
# #     top1_accuracy = accuracy_score(all_labels, all_preds)
# #
# #     # Top-5 准确率
# #     top5_preds = torch.topk(torch.tensor(all_scores), 5, dim=1).indices.numpy()
# #     top5_accuracy = sum(1 for i, label in enumerate(all_labels) if label in top5_preds[i]) / len(all_labels)
# #
# #     # AUC
# #     all_scores = torch.tensor(all_scores)
# #     all_labels = torch.tensor(all_labels)
# #     auc_score = roc_auc_score(all_labels, all_scores, multi_class='ovr')
# #
# #     # 记录评价指标到日志文件
# #     if log_file:
# #         with open(log_file, 'a') as f:
# #             f.write(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}\n')
# #             f.write(f'Top-1 准确率: {top1_accuracy:.4f}\n')
# #             f.write(f'Top-5 准确率: {top5_accuracy:.4f}\n')
# #             f.write(f'AUC: {auc_score:.4f}\n')
# #             f.write(f'平均损失: {np.mean(losses):.4f}\n')
# #             f.write('混淆矩阵:\n')
# #             f.write(f'{cm}\n')
# #             f.write('分类报告:\n')
# #             for cls in cls_report:
# #                 if isinstance(cls_report[cls], dict):
# #                     f.write(f'\n类别: {cls}\n')
# #                     f.write(f"精确率: {cls_report[cls]['precision']:.4f}\n")
# #                     f.write(f"召回率: {cls_report[cls]['recall']:.4f}\n")
# #                     f.write(f"F1值: {cls_report[cls]['f1-score']:.4f}\n")
# #                     f.write(f"支持数: {cls_report[cls]['support']}\n")
# #             f.write('\n')
# #
# #     # 可视化评价指标
# #     if save_dir:
# #         if not os.path.exists(save_dir):
# #             os.makedirs(save_dir)
# #
# #         # 损失值
# #         plt.figure()
# #         plt.plot(losses, label='Loss')
# #         plt.xlabel('Batch')
# #         plt.ylabel('Loss')
# #         plt.title('Evaluation Loss')
# #         plt.legend()
# #         plt.savefig(os.path.join(save_dir, 'evaluation_loss.png'))
# #         plt.close()
# #
# #         # 准确率
# #         plt.figure()
# #         plt.bar(['Accuracy'], [accuracy], color='blue')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation Accuracy')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_accuracy.png'))
# #         plt.close()
# #
# #         # 精确率
# #         plt.figure()
# #         plt.bar(['Precision'], [precision], color='green')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation Precision')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_precision.png'))
# #         plt.close()
# #
# #         # 召回率
# #         plt.figure()
# #         plt.bar(['Recall'], [recall], color='red')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation Recall')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_recall.png'))
# #         plt.close()
# #
# #         # F1值
# #         plt.figure()
# #         plt.bar(['F1 Score'], [f1], color='purple')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation F1 Score')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_f1_score.png'))
# #         plt.close()
# #
# #         # Top-1 准确率
# #         plt.figure()
# #         plt.bar(['Top-1 Accuracy'], [top1_accuracy], color='orange')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation Top-1 Accuracy')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_top1_accuracy.png'))
# #         plt.close()
# #
# #         # Top-5 准确率
# #         plt.figure()
# #         plt.bar(['Top-5 Accuracy'], [top5_accuracy], color='cyan')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation Top-5 Accuracy')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_top5_accuracy.png'))
# #         plt.close()
# #
# #         # AUC
# #         plt.figure()
# #         plt.bar(['AUC'], [auc_score], color='magenta')
# #         plt.ylabel('Score')
# #         plt.title('Evaluation AUC')
# #         plt.savefig(os.path.join(save_dir, 'evaluation_auc.png'))
# #         plt.close()
# #
# #     return accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses
# def evaluate_model(model, dataloader, class_indict, device, log_file=None, save_dir=None):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_scores = []
#     losses = []
#
#     with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:  # 设置进度条总长度
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
#
#             loss = torch.nn.functional.cross_entropy(outputs, labels)
#             losses.append(loss.item())
#
#             progress_bar.update(1)  # 更新进度条
#
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     f1 = f1_score(all_labels, all_preds, average='weighted')
#
#     # 混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#
#     # 分类报告
#     cls_report = classification_report(all_labels, all_preds, target_names=list(class_indict.values()),
#                                        output_dict=True)
#
#     # Top-1 准确率
#     top1_accuracy = accuracy_score(all_labels, all_preds)
#
#     # Top-5 准确率 (修改此部分)
#     num_classes = len(class_indict)
#     all_scores_array = np.array(all_scores)  # 先转换为numpy数组提高效率
#
#     if num_classes >= 5:
#         # 类别数大于等于5时计算Top-5准确率
#         top5_preds = torch.topk(torch.from_numpy(all_scores_array), 5, dim=1).indices.numpy()
#         top5_accuracy = sum(1 for i, label in enumerate(all_labels) if label in top5_preds[i]) / len(all_labels)
#     else:
#         # 类别数少于5时，Top-5准确率与Top-1相同
#         top5_accuracy = top1_accuracy
#
#     # AUC (修改此部分，添加类别数检查)
#     all_labels_tensor = torch.tensor(all_labels)
#     all_scores_tensor = torch.from_numpy(all_scores_array)
#
#     if num_classes > 1:  # 多分类情况
#         if num_classes == 2:  # 二分类
#             auc_score = roc_auc_score(all_labels, all_scores_array[:, 1])
#         else:  # 多分类
#             auc_score = roc_auc_score(all_labels_tensor, all_scores_tensor, multi_class='ovr')
#     else:
#         auc_score = 0.0  # 单类别无法计算AUC
#
#     # 记录评价指标到日志文件
#     if log_file:
#         with open(log_file, 'a') as f:
#             f.write(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}\n')
#             f.write(f'Top-1 准确率: {top1_accuracy:.4f}\n')
#             f.write(f'Top-5 准确率: {top5_accuracy:.4f}\n')
#             f.write(f'AUC: {auc_score:.4f}\n')
#             f.write(f'平均损失: {np.mean(losses):.4f}\n')
#             f.write('混淆矩阵:\n')
#             f.write(f'{cm}\n')
#             f.write('分类报告:\n')
#             for cls in cls_report:
#                 if isinstance(cls_report[cls], dict):
#                     f.write(f'\n类别: {cls}\n')
#                     f.write(f"精确率: {cls_report[cls]['precision']:.4f}\n")
#                     f.write(f"召回率: {cls_report[cls]['recall']:.4f}\n")
#                     f.write(f"F1值: {cls_report[cls]['f1-score']:.4f}\n")
#                     f.write(f"支持数: {cls_report[cls]['support']}\n")
#             f.write('\n')
#
#     # 可视化评价指标
#     if save_dir:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         # 损失值
#         plt.figure()
#         plt.plot(losses, label='Loss')
#         plt.xlabel('Batch')
#         plt.ylabel('Loss')
#         plt.title('Evaluation Loss')
#         plt.legend()
#         plt.savefig(os.path.join(save_dir, 'evaluation_loss.png'))
#         plt.close()
#
#         # 准确率
#         plt.figure()
#         plt.bar(['Accuracy'], [accuracy], color='blue')
#         plt.ylabel('Score')
#         plt.title('Evaluation Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_accuracy.png'))
#         plt.close()
#
#         # 精确率
#         plt.figure()
#         plt.bar(['Precision'], [precision], color='green')
#         plt.ylabel('Score')
#         plt.title('Evaluation Precision')
#         plt.savefig(os.path.join(save_dir, 'evaluation_precision.png'))
#         plt.close()
#
#         # 召回率
#         plt.figure()
#         plt.bar(['Recall'], [recall], color='red')
#         plt.ylabel('Score')
#         plt.title('Evaluation Recall')
#         plt.savefig(os.path.join(save_dir, 'evaluation_recall.png'))
#         plt.close()
#
#         # F1值
#         plt.figure()
#         plt.bar(['F1 Score'], [f1], color='purple')
#         plt.ylabel('Score')
#         plt.title('Evaluation F1 Score')
#         plt.savefig(os.path.join(save_dir, 'evaluation_f1_score.png'))
#         plt.close()
#
#         # Top-1 准确率
#         plt.figure()
#         plt.bar(['Top-1 Accuracy'], [top1_accuracy], color='orange')
#         plt.ylabel('Score')
#         plt.title('Evaluation Top-1 Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_top1_accuracy.png'))
#         plt.close()
#
#         # Top-5 准确率
#         plt.figure()
#         plt.bar(['Top-5 Accuracy'], [top5_accuracy], color='cyan')
#         plt.ylabel('Score')
#         plt.title('Evaluation Top-5 Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_top5_accuracy.png'))
#         plt.close()
#
#         # AUC
#         plt.figure()
#         plt.bar(['AUC'], [auc_score], color='magenta')
#         plt.ylabel('Score')
#         plt.title('Evaluation AUC')
#         plt.savefig(os.path.join(save_dir, 'evaluation_auc.png'))
#         plt.close()
#
#     return accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses
#
# def main():
#     random_seed = 50
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     img_size = 224
#     data_transform = transforms.Compose([
#         transforms.Resize(int(img_size * 1.14)),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # 载入数据集
#     dataset_path = "D:\\CQUBPMDD\\test"  # 数据集路径
#     dataset = ImageFolder(dataset_path, transform=data_transform)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
#
#     # 读取类别标签
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), "文件 '{}' 不存在.".format(json_path)
#     with open(json_path, "r") as f:
#         class_indict = json.load(f)
#
#     # 创建模型
#     model = create_model(num_classes=len(class_indict)).to(device)
#
#     # 载入模型权重
#     model_weight_path = "G:\swin0811\sccov+fca-87.67.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
#
#     # 设置日志记录文件
#     log_file = 'evaluation.log'
#     setup_logging(log_file)
#
#     # 设置保存目录
#     save_dir = 'evaluation_results'
#
#     # 评估模型
#     accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses = evaluate_model(
#         model, dataloader, class_indict, device, log_file=log_file, save_dir=save_dir)
#
#     # 输出评估结果
#     print(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}')
#     print(f'Top-1 准确率: {top1_accuracy:.4f}')
#     print(f'Top-5 准确率: {top5_accuracy:.4f}')
#     print(f'AUC: {auc_score:.4f}')
#     print('混淆矩阵:')
#     print(cm)
#     print('分类报告:')
#     for cls in cls_report:
#         if isinstance(cls_report[cls], dict):
#             print(f'\n类别: {cls}')
#             print(f"精确率: {cls_report[cls]['precision']:.4f}")
#             print(f"召回率: {cls_report[cls]['recall']:.4f}")
#             print(f"F1值: {cls_report[cls]['f1-score']:.4f}")
#             print(f"支持数: {cls_report[cls]['support']}")
#
#
# if __name__ == '__main__':
#     main()




# import os
# import json
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
#     classification_report, roc_auc_score
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm  # 引入进度条模块
# import random
# import numpy as np
# import logging
# import time
#
# from model import swin_small_patch4_window7_224 as create_model
#
#
# def setup_logging(log_file):
#     """设置日志记录到文件"""
#     logging.basicConfig(filename=log_file,
#                         format='%(asctime)s - %(message)s',
#                         level=logging.INFO)
#
#
# def load_image(img_path, data_transform):
#     assert os.path.exists(img_path), "文件 '{}' 不存在.".format(img_path)
#     img = Image.open(img_path).convert('RGB')
#     img = data_transform(img)
#     return img
#
#
# def count_parameters(model):
#     """计算模型参数数量"""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# def calculate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
#     """计算模型的FLOPs"""
#     try:
#         from thop import profile
#         dummy_input = torch.randn(input_size).to(device)
#         flops, params = profile(model, inputs=(dummy_input,))
#         return flops, params
#     except ImportError:
#         print("未安装thop库，无法计算FLOPs")
#         return None, None
#
#
# def measure_inference_speed(model, input_size=(1, 3, 224, 224), device='cpu', num_runs=100):
#     """测量模型推理速度"""
#     dummy_input = torch.randn(input_size).to(device)
#     model.eval()
#
#     # 预热
#     with torch.no_grad():
#         for _ in range(10):
#             _ = model(dummy_input)
#
#     # 实际测量
#     start_time = time.time()
#     with torch.no_grad():
#         for _ in range(num_runs):
#             _ = model(dummy_input)
#     end_time = time.time()
#
#     total_time = end_time - start_time
#     avg_time = total_time / num_runs
#     fps = num_runs / total_time
#
#     return avg_time, fps
#
#
# def evaluate_model(model, dataloader, class_indict, device, log_file=None, save_dir=None):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_scores = []
#     losses = []
#
#     with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:  # 设置进度条总长度
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())
#
#             loss = torch.nn.functional.cross_entropy(outputs, labels)
#             losses.append(loss.item())
#
#             progress_bar.update(1)  # 更新进度条
#
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     f1 = f1_score(all_labels, all_preds, average='weighted')
#
#     # 混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#
#     # 分类报告
#     cls_report = classification_report(all_labels, all_preds, target_names=list(class_indict.values()),
#                                        output_dict=True)
#
#     # Top-1 准确率
#     top1_accuracy = accuracy_score(all_labels, all_preds)
#
#     # Top-5 准确率 (修改此部分)
#     num_classes = len(class_indict)
#     all_scores_array = np.array(all_scores)  # 先转换为numpy数组提高效率
#
#     if num_classes >= 5:
#         # 类别数大于等于5时计算Top-5准确率
#         top5_preds = torch.topk(torch.from_numpy(all_scores_array), 5, dim=1).indices.numpy()
#         top5_accuracy = sum(1 for i, label in enumerate(all_labels) if label in top5_preds[i]) / len(all_labels)
#     else:
#         # 类别数少于5时，Top-5准确率与Top-1相同
#         top5_accuracy = top1_accuracy
#
#     # AUC (修改此部分，添加类别数检查)
#     all_labels_tensor = torch.tensor(all_labels)
#     all_scores_tensor = torch.from_numpy(all_scores_array)
#
#     if num_classes > 1:  # 多分类情况
#         if num_classes == 2:  # 二分类
#             auc_score = roc_auc_score(all_labels, all_scores_array[:, 1])
#         else:  # 多分类
#             auc_score = roc_auc_score(all_labels_tensor, all_scores_tensor, multi_class='ovr')
#     else:
#         auc_score = 0.0  # 单类别无法计算AUC
#
#     # 记录评价指标到日志文件
#     if log_file:
#         with open(log_file, 'a') as f:
#             f.write(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}\n')
#             f.write(f'Top-1 准确率: {top1_accuracy:.4f}\n')
#             f.write(f'Top-5 准确率: {top5_accuracy:.4f}\n')
#             f.write(f'AUC: {auc_score:.4f}\n')
#             f.write(f'平均损失: {np.mean(losses):.4f}\n')
#             f.write('混淆矩阵:\n')
#             f.write(f'{cm}\n')
#             f.write('分类报告:\n')
#             for cls in cls_report:
#                 if isinstance(cls_report[cls], dict):
#                     f.write(f'\n类别: {cls}\n')
#                     f.write(f"精确率: {cls_report[cls]['precision']:.4f}\n")
#                     f.write(f"召回率: {cls_report[cls]['recall']:.4f}\n")
#                     f.write(f"F1值: {cls_report[cls]['f1-score']:.4f}\n")
#                     f.write(f"支持数: {cls_report[cls]['support']}\n")
#             f.write('\n')
#
#     # 可视化评价指标
#     if save_dir:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         # 损失值
#         plt.figure()
#         plt.plot(losses, label='Loss')
#         plt.xlabel('Batch')
#         plt.ylabel('Loss')
#         plt.title('Evaluation Loss')
#         plt.legend()
#         plt.savefig(os.path.join(save_dir, 'evaluation_loss.png'))
#         plt.close()
#
#         # 准确率
#         plt.figure()
#         plt.bar(['Accuracy'], [accuracy], color='blue')
#         plt.ylabel('Score')
#         plt.title('Evaluation Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_accuracy.png'))
#         plt.close()
#
#         # 精确率
#         plt.figure()
#         plt.bar(['Precision'], [precision], color='green')
#         plt.ylabel('Score')
#         plt.title('Evaluation Precision')
#         plt.savefig(os.path.join(save_dir, 'evaluation_precision.png'))
#         plt.close()
#
#         # 召回率
#         plt.figure()
#         plt.bar(['Recall'], [recall], color='red')
#         plt.ylabel('Score')
#         plt.title('Evaluation Recall')
#         plt.savefig(os.path.join(save_dir, 'evaluation_recall.png'))
#         plt.close()
#
#         # F1值
#         plt.figure()
#         plt.bar(['F1 Score'], [f1], color='purple')
#         plt.ylabel('Score')
#         plt.title('Evaluation F1 Score')
#         plt.savefig(os.path.join(save_dir, 'evaluation_f1_score.png'))
#         plt.close()
#
#         # Top-1 准确率
#         plt.figure()
#         plt.bar(['Top-1 Accuracy'], [top1_accuracy], color='orange')
#         plt.ylabel('Score')
#         plt.title('Evaluation Top-1 Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_top1_accuracy.png'))
#         plt.close()
#
#         # Top-5 准确率
#         plt.figure()
#         plt.bar(['Top-5 Accuracy'], [top5_accuracy], color='cyan')
#         plt.ylabel('Score')
#         plt.title('Evaluation Top-5 Accuracy')
#         plt.savefig(os.path.join(save_dir, 'evaluation_top5_accuracy.png'))
#         plt.close()
#
#         # AUC
#         plt.figure()
#         plt.bar(['AUC'], [auc_score], color='magenta')
#         plt.ylabel('Score')
#         plt.title('Evaluation AUC')
#         plt.savefig(os.path.join(save_dir, 'evaluation_auc.png'))
#         plt.close()
#
#     return accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses
#
#
# def main():
#     random_seed = 50
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     img_size = 224
#     data_transform = transforms.Compose([
#         transforms.Resize(int(img_size * 1.14)),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # 载入数据集
#     dataset_path = "D:\\CQUBPMDD\\test"  # 数据集路径
#     dataset = ImageFolder(dataset_path, transform=data_transform)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
#
#     # 读取类别标签
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), "文件 '{}' 不存在.".format(json_path)
#     with open(json_path, "r") as f:
#         class_indict = json.load(f)
#
#     # 创建模型
#     model = create_model(num_classes=len(class_indict)).to(device)
#
#     # 载入模型权重
#     model_weight_path = 'G:\swin0811\sccov+fca-87.67.pth'
#     model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
#
#     # 设置日志记录文件
#     log_file = 'evaluation.log'
#     setup_logging(log_file)
#
#     # 设置保存目录
#     save_dir = 'evaluation_results'
#
#     # 计算模型参数数量
#     num_params = count_parameters(model)
#     print(f"模型参数数量: {num_params:,}")
#
#     # 计算模型FLOPs
#     flops, _ = calculate_flops(model, input_size=(1, 3, 224, 224), device=device)
#     if flops is not None:
#         print(f"模型FLOPs: {flops/1e9:.2f}G")
#
#     # 测量推理速度
#     inference_time, inference_fps = measure_inference_speed(model, input_size=(1, 3, 224, 224), device=device)
#     print(f"单张图像推理时间: {inference_time*1000:.2f}ms, FPS: {inference_fps:.2f}")
#
#     # 评估模型
#     accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses = evaluate_model(
#         model, dataloader, class_indict, device, log_file=log_file, save_dir=save_dir)
#
#     # 输出评估结果
#     print(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}')
#     print(f'Top-1 准确率: {top1_accuracy:.4f}')
#     print(f'Top-5 准确率: {top5_accuracy:.4f}')
#     print(f'AUC: {auc_score:.4f}')
#     print('混淆矩阵:')
#     print(cm)
#     print('分类报告:')
#     for cls in cls_report:
#         if isinstance(cls_report[cls], dict):
#             print(f'\n类别: {cls}')
#             print(f"精确率: {cls_report[cls]['precision']:.4f}")
#             print(f"召回率: {cls_report[cls]['recall']:.4f}")
#             print(f"F1值: {cls_report[cls]['f1-score']:.4f}")
#             print(f"支持数: {cls_report[cls]['support']}")
#
#     # 输出效率指标到日志文件
#     with open(log_file, 'a') as f:
#         f.write(f"\n效率指标:\n")
#         f.write(f"模型参数数量: {num_params:,}\n")
#         if flops is not None:
#             f.write(f"模型FLOPs: {flops/1e9:.2f}G\n")
#         f.write(f"单张图像推理时间: {inference_time*1000:.2f}ms\n")
#         f.write(f"FPS: {inference_fps:.2f}\n")
#
#
# if __name__ == '__main__':
#     main()
import os
import json
import torch
from PIL import Image
 # 忽略过时模块的警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 使用新的transforms接口
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import random
import numpy as np
import logging
import time

from model import swin_small_patch4_window7_224 as create_model


def setup_logging(log_file):
    """设置日志记录到文件"""
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)


def load_image(img_path, data_transform):
    assert os.path.exists(img_path), "文件 '{}' 不存在.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = data_transform(img)
    return img


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """计算模型的FLOPs"""
    try:
        from thop import profile
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,))
        return flops, params
    except ImportError:
        print("未安装thop库，无法计算FLOPs")
        return None, None


def measure_inference_speed(model, input_size=(1, 3, 224, 224), device='cpu', num_runs=100):
    """测量模型推理速度"""
    dummy_input = torch.randn(input_size).to(device)
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 实际测量
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time

    return avg_time, fps


def evaluate_model(model, dataloader, class_indict, device, log_file=None, save_dir=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    losses = []

    with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            losses.append(loss.item())

            progress_bar.update(1)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 分类报告
    cls_report = classification_report(all_labels, all_preds, target_names=list(class_indict.values()),
                                       output_dict=True)

    # Top-1 准确率
    top1_accuracy = accuracy_score(all_labels, all_preds)

    # Top-5 准确率
    num_classes = len(class_indict)
    all_scores_array = np.array(all_scores)

    if num_classes >= 5:
        # 类别数大于等于5时计算Top-5准确率
        top5_preds = torch.topk(torch.from_numpy(all_scores_array), 5, dim=1).indices.numpy()
        top5_accuracy = sum(1 for i, label in enumerate(all_labels) if label in top5_preds[i]) / len(all_labels)
    else:
        # 类别数少于5时，Top-5准确率与Top-1相同
        top5_accuracy = top1_accuracy

    # AUC
    all_labels_tensor = torch.tensor(all_labels)
    all_scores_tensor = torch.from_numpy(all_scores_array)

    if num_classes > 1:  # 多分类情况
        if num_classes == 2:  # 二分类
            auc_score = roc_auc_score(all_labels, all_scores_array[:, 1])
        else:  # 多分类
            auc_score = roc_auc_score(all_labels_tensor, all_scores_tensor, multi_class='ovr')
    else:
        auc_score = 0.0  # 单类别无法计算AUC

    # 记录评价指标到日志文件
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}\n')
            f.write(f'Top-1 准确率: {top1_accuracy:.4f}\n')
            f.write(f'Top-5 准确率: {top5_accuracy:.4f}\n')
            f.write(f'AUC: {auc_score:.4f}\n')
            f.write(f'平均损失: {np.mean(losses):.4f}\n')
            f.write('混淆矩阵:\n')
            f.write(f'{cm}\n')
            f.write('分类报告:\n')
            for cls in cls_report:
                if isinstance(cls_report[cls], dict):
                    f.write(f'\n类别: {cls}\n')
                    f.write(f"精确率: {cls_report[cls]['precision']:.4f}\n")
                    f.write(f"召回率: {cls_report[cls]['recall']:.4f}\n")
                    f.write(f"F1值: {cls_report[cls]['f1-score']:.4f}\n")
                    f.write(f"支持数: {cls_report[cls]['support']}\n")
            f.write('\n')

    # 可视化评价指标
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 损失值
        plt.figure()
        plt.plot(losses, label='Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'evaluation_loss.png'))
        plt.close()

        # 准确率
        plt.figure()
        plt.bar(['Accuracy'], [accuracy], color='blue')
        plt.ylabel('Score')
        plt.title('Evaluation Accuracy')
        plt.savefig(os.path.join(save_dir, 'evaluation_accuracy.png'))
        plt.close()

        # 精确率
        plt.figure()
        plt.bar(['Precision'], [precision], color='green')
        plt.ylabel('Score')
        plt.title('Evaluation Precision')
        plt.savefig(os.path.join(save_dir, 'evaluation_precision.png'))
        plt.close()

        # 召回率
        plt.figure()
        plt.bar(['Recall'], [recall], color='red')
        plt.ylabel('Score')
        plt.title('Evaluation Recall')
        plt.savefig(os.path.join(save_dir, 'evaluation_recall.png'))
        plt.close()

        # F1值
        plt.figure()
        plt.bar(['F1 Score'], [f1], color='purple')
        plt.ylabel('Score')
        plt.title('Evaluation F1 Score')
        plt.savefig(os.path.join(save_dir, 'evaluation_f1_score.png'))
        plt.close()

        # Top-1 准确率
        plt.figure()
        plt.bar(['Top-1 Accuracy'], [top1_accuracy], color='orange')
        plt.ylabel('Score')
        plt.title('Evaluation Top-1 Accuracy')
        plt.savefig(os.path.join(save_dir, 'evaluation_top1_accuracy.png'))
        plt.close()

        # Top-5 准确率
        plt.figure()
        plt.bar(['Top-5 Accuracy'], [top5_accuracy], color='cyan')
        plt.ylabel('Score')
        plt.title('Evaluation Top-5 Accuracy')
        plt.savefig(os.path.join(save_dir, 'evaluation_top5_accuracy.png'))
        plt.close()

        # AUC
        plt.figure()
        plt.bar(['AUC'], [auc_score], color='magenta')
        plt.ylabel('Score')
        plt.title('Evaluation AUC')
        plt.savefig(os.path.join(save_dir, 'evaluation_auc.png'))
        plt.close()

    return accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses


def main():
    random_seed = 50
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 载入数据集
    dataset_path = "D:\\CQUBPMDD\\val" # 数据集路径
    dataset = ImageFolder(dataset_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 读取类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "文件 '{}' 不存在.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=len(class_indict)).to(device)


    # 载入模型权重 - 修复路径问题
    model_weight_path = "G:\\swin0811\\bpmdd-mab+fca=88.4.pth"  # 修改为相对路径
    if not os.path.exists(model_weight_path):
        # 如果相对路径不存在，尝试使用绝对路径
        model_weight_path = "G:\\swin0811\\bpmdd-mab+fca=88.4.pth"

    if not os.path.exists(model_weight_path):
        # 如果权重文件不存在，给出提示
        print(f"警告：模型权重文件 '{model_weight_path}' 不存在")
        print("请确保模型权重文件存在于指定路径，或修改代码中的路径")
        return

    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

    # 设置日志记录文件
    log_file = 'evaluation.log'
    setup_logging(log_file)

    # 设置保存目录
    save_dir = 'evaluation_results'

    # 计算模型参数数量
    num_params = count_parameters(model)
    print(f"模型参数数量: {num_params:,}")

    # 计算模型FLOPs
    flops, _ = calculate_flops(model, input_size=(1, 3, 224, 224), device=device)
    if flops is not None:
        print(f"模型FLOPs: {flops/1e9:.2f}G")

    # 测量推理速度
    inference_time, inference_fps = measure_inference_speed(model, input_size=(1, 3, 224, 224), device=device)
    print(f"单张图像推理时间: {inference_time*1000:.2f}ms, FPS: {inference_fps:.2f}")

    # 评估模型
    accuracy, precision, recall, f1, cm, cls_report, top1_accuracy, top5_accuracy, auc_score, losses = evaluate_model(
        model, dataloader, class_indict, device, log_file=log_file, save_dir=save_dir)

    # 输出评估结果
    print(f'准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}')
    print(f'Top-1 准确率: {top1_accuracy:.4f}')
    print(f'Top-5 准确率: {top5_accuracy:.4f}')
    print(f'AUC: {auc_score:.4f}')
    print('混淆矩阵:')
    print(cm)
    print('分类报告:')
    for cls in cls_report:
        if isinstance(cls_report[cls], dict):
            print(f'\n类别: {cls}')
            print(f"精确率: {cls_report[cls]['precision']:.4f}")
            print(f"召回率: {cls_report[cls]['recall']:.4f}")
            print(f"F1值: {cls_report[cls]['f1-score']:.4f}")
            print(f"支持数: {cls_report[cls]['support']}")

    # 输出效率指标到日志文件
    with open(log_file, 'a') as f:
        f.write(f"\n效率指标:\n")
        f.write(f"模型参数数量: {num_params:,}\n")
        if flops is not None:
            f.write(f"模型FLOPs: {flops/1e9:.2f}G\n")
        f.write(f"单张图像推理时间: {inference_time*1000:.2f}ms\n")
        f.write(f"FPS: {inference_fps:.2f}\n")


if __name__ == '__main__':
    main()
