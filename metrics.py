import numpy as np
import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def recall_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间
    output = torch.sigmoid(output)

    # 将输出和目标展平为一维数组
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例数目
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # 计算真实的正例数目
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))

    # 计算召回率
    recall = true_positives / (actual_positives + smooth)

    return recall


def precision_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间
    output = torch.sigmoid(output)

    # 将输出和目标展平为一维数组
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例数目
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))

    # 计算预测的正例数目
    predicted_positives = np.sum(np.round(np.clip(output, 0, 1)))

    # 计算精确度
    precision = true_positives / (predicted_positives + smooth)

    return precision


def accuracy_s(output, target):
    # 将输出用sigmoid函数压缩到0到1之间，并四舍五入为0或1
    predicted = torch.round(torch.sigmoid(output))

    # 将预测值和目标值转换为CPU上的numpy数组
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算准确度
    accuracy = np.mean(predicted == target)

    return accuracy


def jaccard_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)
def f1_score_s(output, target):
    smooth = 1e-5

    # 计算精确度
    precision = precision_s(output, target)

    # 计算召回率
    recall = recall_s(output, target)

    # 计算 F1 score
    f1 = (2 * precision * recall) / (precision + recall + smooth)

    return f1
def specificity_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间，并四舍五入为0或1
    predicted = torch.round(torch.sigmoid(output))

    # 将预测值和目标值转换为CPU上的numpy数组
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真负例数目
    true_negatives = np.sum(np.round(np.clip((1 - predicted) * (1 - target), 0, 1)))

    # 计算真实的负例数目
    actual_negatives = np.sum(np.round(np.clip(1 - target, 0, 1)))

    # 计算特异性
    specificity = true_negatives / (actual_negatives + smooth)

    return specificity

def sensitivity_s(output, target):
    smooth = 1e-5

    # 将输出用sigmoid函数压缩到0到1之间，并四舍五入为0或1
    predicted = torch.round(torch.sigmoid(output))

    # 将预测值和目标值转换为CPU上的numpy数组
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()

    # 计算真正例数目
    true_positives = np.sum(np.round(np.clip(predicted * target, 0, 1)))

    # 计算真实的正例数目
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))

    # 计算敏感性
    sensitivity = true_positives / (actual_positives + smooth)

    return sensitivity


def auc_score(output, target):
    # 强制二值化目标标签（假设原始标签为0/255或连续值）
    target = (target > 0).float()

    # 转换为概率并展平
    output_prob = torch.sigmoid(output).squeeze(1).view(-1).cpu().numpy()
    target = target.squeeze(1).view(-1).cpu().numpy()

    # 处理多标签情况（若为单标签，直接计算）
    if len(target.shape) > 1 and target.shape[1] > 1:
        # 多标签：设置 average='micro' 或 'macro'
        auc = roc_auc_score(target, output_prob, average='micro')
    else:
        # 单标签二分类
        auc = roc_auc_score(target, output_prob)

    return auc

def mcc_score(output, target):
    # 确保目标标签为二值标签（假设目标标签是 0 到 255 范围的值）
    target = (target > 0).float()  # 将目标标签转换为 0 和 1 的二值标签

    # 将模型输出转换为概率值
    output_prob = torch.sigmoid(output)

    # 将概率值转换为二值标签
    predicted = (output_prob > 0.5).float()  # 使用 0.5 作为阈值

    # 将张量转换为 numpy 数组
    target_array = target.squeeze(1).view(-1).cpu().numpy()
    predicted = predicted.squeeze(1).view(-1).cpu().numpy()

    # 计算马修斯相关系数
    mcc = matthews_corrcoef(target_array, predicted)
    return mcc

