import os
import json
import torch
import numpy as np
import collections
from dataset.datasets import Data
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net_backbone, dataset_path, batch_size=128, num_workers=2, lr=0.001, weight_decay=1e-6):
    # 加载数据集
    print(dataset_path)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Data(dataset_path + '/train/', transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            drop_last=False)

    # 优化器和损失函数配置
    optimizer = torch.optim.Adam(net_backbone.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()

    g_total= 0
    g_correct = 0
    g_loss = 0
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)

        # 计算loss和acc
        prediction = net_backbone(img)
        loss = loss_func(prediction, label)
        g_loss += loss.item()
        print("train loss: {:.4}".format(loss.item()))
        prediction = torch.argmax(prediction, 1)
        correct = (prediction == label).sum().float()
        print("train batch acc: {:.4}".format(correct/len(label)))
        g_correct += correct
        g_total += len(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("train loss: {:.4}".format(g_loss))
    print("train acc: {:.4}".format(g_correct/g_total))
    return g_correct/g_total


def validate_model(val_dataloader, net_backbone, loss_func):
    losses = []
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            # label = label.to(device).type(torch.float32)
            label = label.to(device)

            prediction = net_backbone(img)
            loss = loss_func(prediction, label)
            losses.append(loss)
        print("avg val loss: {:.4}".format(np.mean(losses)))
    return np.mean(losses)


def save_model(eye_backbone, mouth_backbone, args_file_path, eye_pth_path, mouth_pth_path):
    # 模型参数写入文件
    global_weights_list = {
        "eye": {},
        "mouth": {}
    }
    for key, value in eye_backbone.state_dict().items():
        global_weights_list["eye"][key] = value.cpu().numpy().tolist()
    for key, value in mouth_backbone.state_dict().items():
        global_weights_list["mouth"][key] = value.cpu().numpy().tolist()

    f = open(args_file_path, "w")
    f.write(json.dumps(global_weights_list))
    f.close()
    print("save model args success! path:" + args_file_path)

    # Torch 模型写入文件
    torch.save(eye_backbone, eye_pth_path)
    print("save eye   model pth success! path:" + eye_pth_path)
    torch.save(mouth_backbone, mouth_pth_path)
    print("save mouth model pth success! path:" + mouth_pth_path)
    return True


def load_model_from_args(eye_backbone, mouth_backbone, args_file_path):
    if not os.path.exists(args_file_path):
        print(f"args path {args_file_path} not exist!")
        return False
    f = open(args_file_path, "r")
    global_weight_str = f.read()
    f.close()
    global_weight = json.loads(global_weight_str)

    for key, value in global_weight["eye"].items():
        global_weight["eye"][key] = torch.tensor(value)
    for key, value in global_weight["mouth"].items():
        global_weight["mouth"][key] = torch.tensor(value)
    eye_backbone.load_state_dict(collections.OrderedDict(global_weight["eye"]))
    mouth_backbone.load_state_dict(collections.OrderedDict(global_weight["mouth"]))
    print("load model args success! path:" + args_file_path)


def model_to_onnx(torch_model_path, onnx_model_path):
    # pytorch模型加载
    torch_model = torch.load(torch_model_path)
    # set the mouth_model to inference mode
    torch_model.eval()
    torch_model.to(device)

    # 生成输入格式并导出模型
    x = torch.randn(1, 3, 64, 64).to(device)
    torch.onnx.export(torch_model,
                        x,
                        onnx_model_path,
                        input_names=["input"], # 输入名
                        output_names=["output"], # 输出名
                        verbose=False,
                        opset_version=12,
                    )