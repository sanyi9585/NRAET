from my_dataset import MyDataSet
from model import swin_small_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate, count_parameters, calculate_flops, measure_inference_speed
import os
import torch
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def main(args):
#     set_seed(args.seed)
#
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     if not os.path.exists("./weights"):
#         os.makedirs("./weights")
#
#     tb_writer = SummaryWriter()
#
#     # 从指定的文件夹读取训练集和验证集数据
#     train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
#         args.train_data_path, args.val_data_path
#     )
#     img_size = 224
#     data_transform = {
#         "train": transforms.Compose([
#             transforms.RandomResizedCrop(img_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         "val": transforms.Compose([
#             transforms.Resize(int(img_size * 1.143)),
#             transforms.CenterCrop(img_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }
#
#     # 创建训练集和验证集的数据集对象
#     train_dataset = MyDataSet(images_path=train_images_path,
#                               images_class=train_images_label,
#                               transform=data_transform["train"])
#
#     val_dataset = MyDataSet(images_path=val_images_path,
#                             images_class=val_images_label,
#                             transform=data_transform["val"])
#
#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#     print(f'Using {nw} dataloader workers every process')
#
#     # 创建训练集和验证集的数据加载器
#     train_loader = DataLoader(train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               pin_memory=True,
#                               num_workers=nw,
#                               collate_fn=train_dataset.collate_fn)
#
#     val_loader = DataLoader(val_dataset,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             pin_memory=True,
#                             num_workers=nw,
#                             collate_fn=val_dataset.collate_fn)
#
#     # 创建模型并将其移动到指定的设备上
#     model = create_model(num_classes=args.num_classes).to(device)
#
#     # 如果指定了预训练权重，则加载权重，删除与模型结构不匹配的部分
#     if args.weights != "":
#         assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
#         weights_dict = torch.load(args.weights, map_location=device)["model"]
#         for k in list(weights_dict.keys()):
#             if "head" in k:
#                 del weights_dict[k]
#         model.load_state_dict(weights_dict, strict=False)
#
#     # 如果需要冻结部分层，则冻结除了头部之外的所有参数
#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             if "head" not in name:
#                 para.requires_grad_(False)
#             else:
#                 print(f"training {name}")
#
#     # 获取可训练的参数列表，并创建优化器
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.0001)
#
#     best_val_acc = 0.0  # 记录最佳验证集精度
#     best_epoch = 0  # 记录最佳模型的epoch
#
#     # 训练循环
#     for epoch in range(args.epochs):
#         # 训练一个epoch并获取训练集的指标
#         train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_one_epoch(
#             model=model,
#             optimizer=optimizer,
#             data_loader=train_loader,
#             device=device,
#             epoch=epoch
#         )
#
#         # 在验证集上评估模型并获取验证集的指标
#         val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = evaluate(
#             model=model,
#             data_loader=val_loader,
#             device=device,
#             epoch=epoch
#         )
#
#         # 打印当前epoch的训练和验证指标
#         print(f"Epoch {epoch + 1}/{args.epochs}")
#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#               f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
#               f"Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}")
#         print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
#               f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
#               f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
#
#         # 将训练和验证指标写入TensorBoard
#         tags = ["train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_auc",
#                 "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "val_auc", "learning_rate"]
#         tb_writer.add_scalar(tags[0], train_loss, epoch)
#         tb_writer.add_scalar(tags[1], train_acc, epoch)
#         tb_writer.add_scalar(tags[2], train_precision, epoch)
#         tb_writer.add_scalar(tags[3], train_recall, epoch)
#         tb_writer.add_scalar(tags[4], train_f1, epoch)
#         tb_writer.add_scalar(tags[5], train_auc, epoch)
#         tb_writer.add_scalar(tags[6], val_loss, epoch)
#         tb_writer.add_scalar(tags[7], val_acc, epoch)
#         tb_writer.add_scalar(tags[8], val_precision, epoch)
#         tb_writer.add_scalar(tags[9], val_recall, epoch)
#         tb_writer.add_scalar(tags[10], val_f1, epoch)
#         tb_writer.add_scalar(tags[11], val_auc, epoch)
#         tb_writer.add_scalar(tags[12], optimizer.param_groups[0]["lr"], epoch)
#
#         # 保存当前模型权重
#         torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")
#
#         # 如果验证集精度提升，则保存当前模型为最佳模型
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_epoch = epoch
#             best_model_path = f"./weights/best_model.pth"
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Saved best model with val_acc: {val_acc:.4f} at epoch {epoch + 1}")
#
#     print(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
def main(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 从指定的文件夹读取训练集和验证集数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.train_data_path, args.val_data_path
    )
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 创建训练集和验证集的数据集对象
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    # 创建模型并将其移动到指定的设备上
    model = create_model(num_classes=args.num_classes).to(device)

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

    # 如果指定了预训练权重，则加载权重，删除与模型结构不匹配的部分
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    # 如果需要冻结部分层，则冻结除了头部之外的所有参数
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    # 获取可训练的参数列表，并创建优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.0001)

    best_val_acc = 0.0  # 记录最佳验证集精度
    best_epoch = 0  # 记录最佳模型的epoch

    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch并获取训练集的指标
        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc, train_time, train_speed = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )

        # 在验证集上评估模型并获取验证集的指标
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_time, val_speed = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # 打印当前epoch的训练和验证指标
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
              f"Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Train Time: {train_time:.2f}s, Train Speed: {train_speed:.2f} samples/s")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Val Time: {val_time:.2f}s, Val Speed: {val_speed:.2f} samples/s")

        # 将训练和验证指标写入TensorBoard
        tags = ["train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "train_auc", "train_time", "train_speed",
                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "val_auc", "val_time", "val_speed", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_precision, epoch)
        tb_writer.add_scalar(tags[3], train_recall, epoch)
        tb_writer.add_scalar(tags[4], train_f1, epoch)
        tb_writer.add_scalar(tags[5], train_auc, epoch)
        tb_writer.add_scalar(tags[6], train_time, epoch)
        tb_writer.add_scalar(tags[7], train_speed, epoch)
        tb_writer.add_scalar(tags[8], val_loss, epoch)
        tb_writer.add_scalar(tags[9], val_acc, epoch)
        tb_writer.add_scalar(tags[10], val_precision, epoch)
        tb_writer.add_scalar(tags[11], val_recall, epoch)
        tb_writer.add_scalar(tags[12], val_f1, epoch)
        tb_writer.add_scalar(tags[13], val_auc, epoch)
        tb_writer.add_scalar(tags[14], val_time, epoch)
        tb_writer.add_scalar(tags[15], val_speed, epoch)
        tb_writer.add_scalar(tags[16], optimizer.param_groups[0]["lr"], epoch)

        # 保存当前模型权重
        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

        # 如果验证集精度提升，则保存当前模型为最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_path = f"./weights/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val_acc: {val_acc:.4f} at epoch {epoch + 1}")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    print(f"模型参数数量: {num_params:,}")
    if flops is not None:
        print(f"模型FLOPs: {flops/1e9:.2f}G")
    print(f"单张图像推理时间: {inference_time*1000:.2f}ms, FPS: {inference_fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00001)

    parser.add_argument('--train-data-path', type=str, default="D:\\Crack500_pdd\\train")
    parser.add_argument('--val-data-path', type=str, default="D:\\Crack500_pdd\\val")

    parser.add_argument('--weights', type=str, default='swin_small_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', action='store_true', default=False,
                        help='Freeze layers except the head during training')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=50, help='random seed')

    opt = parser.parse_args()

    main(opt)
