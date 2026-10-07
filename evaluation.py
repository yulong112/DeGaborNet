import time
from operator import truediv

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from torch import nn

from dataProcess import zeroPadding, window_slides
import json
from torch.cuda.amp import autocast

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f}±{:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f}±{:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f}±{:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f}±{:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)

def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results

def reports(test_loader, y_test, net: nn.Module, name='IP', device='cpu'):
    net.eval()
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        if isinstance(outputs, tuple or list):
            outputs = outputs[0]
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    classification = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def evaluation(test_loader, y_test, net: nn.Module, name='IP', device='cpu',
               save_path="output/classification_report.txt") -> str:
    classification, confusion, oa, each_acc, aa, kappa = reports(test_loader, y_test, net,
                                                                 name, device)
    print('EachA', each_acc)
    print('OA', oa)
    print('AvgA', aa)
    print('kappa', kappa)
    print(classification)
    classification = str(classification)
    confusion = str(confusion)
    with open(save_path, 'w') as x_file:
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
    with open(save_path[:-4] + '.json', 'w', encoding='utf-8') as f:
        json.dump({
            'EachA': {i + 1: each_acc[i] for i in range(len(each_acc))},
            'OA': oa,
            'AA': aa,
            'Kappa': kappa
        }, f, indent=4)
    print('json saved at ' + save_path[:-4] + '.json')
    # json 读取
    # with open("res.json", 'r', encoding='utf-8') as fw:
    #     injson = json.load(fw)
    # print(injson)
    # print(type(injson))
    return '{} Each accuracy (%)'.format(each_acc) + '\n' + \
           '{} Overall accuracy (%)'.format(oa) + '\n' + \
           '{} Average accuracy (%)'.format(aa) + '\n' + \
           '{} Kappa accuracy (%)'.format(kappa) + '\n' + \
           '{}'.format(classification)


def predict(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', is_lstm=False):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    X = zeroPadding(X, patch_size // 2)
    # 逐像素预测类别
    outputs = np.zeros((height, width))
    pre = 0.
    for i in range(height):
        pre_time = 0.
        for j in range(width):
            # if int(y[i, j]) == 0:
                # continue
            # else:
            # for ii in range(1000):
            # if j==50:
            #     print(j)
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            '''
            if is_lstm:
                if dim == 2:
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                      image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 4, 1, 2)).to(device)
                else:
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                      image_patch.shape[2], 1, 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 5, 3, 1, 2)).to(device)
            else:
            '''
            if dim == 3:
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            else:  # dim == 2:
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
            torch.cuda.synchronize(device)
            start_time1 = time.perf_counter()  # 记录开始时间
            with torch.no_grad():
                prediction = net(X_test_image)
            torch.cuda.synchronize(device)
            end_time1 = time.perf_counter()  # 记录开始时间
            pre_time += (end_time1 - start_time1)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
        pre += pre_time
        if i % 50 == 0:
            print('... ... row ', i, ' handling ... ...')
            print(pre)
    print("all time:",pre)
    return outputs.astype(np.uint8)


def predict_optimized(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', batch_size=100):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height, width = y.shape[:2]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width))
    pre_time_total = 0.0  # 累计总时间

    # 预转换输入数据为张量
    X_tensor = torch.FloatTensor(X).to(device)

    # 更大的批量处理，逐行提取图像块
    patches = []
    positions = []
    for i in range(height):
        row_patches = []
        row_positions = []
        for j in range(width):
            image_patch = X_tensor[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.unsqueeze(0)
            if dim == 3:
                image_patch = image_patch.unsqueeze(-1).permute(0, 4, 3, 1, 2)
            else:
                image_patch = image_patch.permute(0, 3, 1, 2)

            row_patches.append(image_patch)
            row_positions.append((i, j))

            # 达到批量大小时进行预测
            if len(row_patches) == batch_size:
                elapsed_time = batch_predict(row_patches, row_positions, outputs, net, device)
                pre_time_total += elapsed_time  # 累计总时间
                row_patches = []
                row_positions = []

        # 处理每行剩余的图像块
        if row_patches:
            elapsed_time = batch_predict(row_patches, row_positions, outputs, net, device)
            pre_time_total += elapsed_time  # 累计总时间

        # 每10行打印累计的总时间
        if (i + 1) % 10 == 0:
            print(f'... ... row {i + 1} handling ... ...')
            print('Total time elapsed so far:', pre_time_total)

    print("Total time:", pre_time_total)
    return outputs.astype(np.uint8)

def batch_predict(patches, positions, outputs, net, device):
    # 使用 CUDA 事件进行批量计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 合并批量
    X_batch = torch.cat(patches).to(device)
    start.record()
    with torch.no_grad():
        predictions = net(X_batch)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000  # 计算批量预测时间

    # 处理预测结果并映射回输出
    predictions = np.argmax(predictions.cpu().numpy(), axis=1)
    for idx, (i, j) in enumerate(positions):
        outputs[i, j] = predictions[idx] + 1

    return elapsed_time  # 返回本批次处理时间


import torch
import numpy as np
from torch import nn


def predict_optimized(X, y, patch_size: int, net: nn.Module, dim=2, device='cuda', batch_size=100):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height, width = y.shape
    X = zeroPadding(X, patch_size // 2)  # 对输入图像进行填充
    outputs = np.zeros((height, width), dtype=np.uint8)
    pre_time_total = 0.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        image_patches = []
        positions = []

        # 逐个处理每行图像块，累积到batch_size大小时进行预测
        for i in range(height):
            for j in range(width):
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                if dim == 3:
                    image_patch = image_patch.reshape(1, *image_patch.shape, 1).transpose(0, 4, 3, 1, 2)
                else:  # dim == 2
                    image_patch = image_patch.reshape(1, *image_patch.shape).transpose(0, 3, 1, 2)

                image_patches.append(image_patch)
                positions.append((i, j))

                # 达到批量大小时进行预测
                if len(image_patches) == batch_size:
                    X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

                    start.record()
                    predictions = net(X_test_image)  # 使用网络进行预测
                    end.record()
                    torch.cuda.synchronize()
                    pre_time_total += start.elapsed_time(end) / 1000

                    # 获取批次预测结果并映射到输出矩阵
                    batch_predictions = np.argmax(predictions.cpu().numpy(), axis=1)
                    for idx, (i_pos, j_pos) in enumerate(positions):
                        outputs[i_pos, j_pos] = batch_predictions[idx] + 1

                    # 清空列表以释放内存，准备下一批次
                    image_patches = []
                    positions = []
            if i % 10 == 0:
                print('... ... row ', i, ' handling ... ...')
                print('Time elapsed:', pre_time_total)
        # 处理最后一批（如果不满batch_size）
        if len(image_patches) > 0:
            X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)
            start.record()
            predictions = net(X_test_image)
            end.record()
            torch.cuda.synchronize()
            pre_time_total += start.elapsed_time(end) / 1000

            batch_predictions = np.argmax(predictions.cpu().numpy(), axis=1)
            for idx, (i_pos, j_pos) in enumerate(positions):
                outputs[i_pos, j_pos] = batch_predictions[idx] + 1

    print("Total inference time:", pre_time_total, "seconds")
    return outputs


def predict1(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', batch_size=10):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width))
    pre_time_total = 0.

    image_patches = []
    positions = []
    desired_batch_size = batch_size  # 您可以根据显存情况调整批量大小
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(height):
        for j in range(width):
            # 提取图像块
            image_patch = X[i:i + patch_size, j:j + patch_size, :]

            # 调整图像块的形状
            if dim == 3:
                image_patch = image_patch.reshape(1, *image_patch.shape, 1)
                processed_patch = image_patch.transpose(0, 4, 3, 1, 2)
            else:  # dim == 2
                image_patch = image_patch.reshape(1, *image_patch.shape)
                processed_patch = image_patch.transpose(0, 3, 1, 2)

            image_patches.append(processed_patch)
            positions.append((i, j))

            # 当达到批量大小时，进行预测
            if len(image_patches) == desired_batch_size:
                X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

                # torch.cuda.synchronize(device)
                # start_time = time.perf_counter()
                start.record()
                with torch.no_grad():
                    predictions = net(X_test_image)

                # torch.cuda.synchronize(device)
                # end_time = time.perf_counter()
                end.record()
                end.synchronize()
                # pre_time_total += (end_time - start_time)
                pre_time_total += (start.elapsed_time(end) / 1000)
                # 处理预测结果并映射回输出
                predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
                for idx, (i_pos, j_pos) in enumerate(positions):
                    outputs[i_pos][j_pos] = predictions[idx] + 1

                # 清空列表，准备下一批次
                image_patches = []
                positions = []

        # 打印进度信息
        if i % 10 == 0:
            print('... ... row ', i, ' handling ... ...')
            print('Time elapsed:', pre_time_total)

    # 处理剩余的图像块
    if len(image_patches) > 0:
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

        # torch.cuda.synchronize(device)
        # start_time = time.perf_counter()
        start.record()
        with torch.no_grad():
            predictions = net(X_test_image)

        # torch.cuda.synchronize(device)
        # end_time = time.perf_counter()
        end.record()
        end.synchronize()
        # pre_time_total += (end_time - start_time)
        pre_time_total += (start.elapsed_time(end) / 1000)
        predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        for idx, (i_pos, j_pos) in enumerate(positions):
            outputs[i_pos][j_pos] = predictions[idx] + 1

    print("Total time:", pre_time_total)
    return outputs.astype(np.uint8)


def quick_predict(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', is_lstm=False,
                  all_process=False, batch_size=512):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    patch_fea, patch_labels = window_slides(X, y, window_size=patch_size, removeZeroLabels=False)
    # 逐像素预测类别
    outputs = np.zeros((height * width,), dtype=np.uint8)
    count_id = 0
    while count_id < outputs.shape[0]:
        image_patch = patch_fea[count_id: count_id + batch_size, ...]
        if is_lstm:
            if dim == 2:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  image_patch.shape[3], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 4, 1, 2)).to(device)
            else:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3], 1, 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 5, 3, 1, 2)).to(device)
        else:
            if dim == 3:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            else:  # dim == 2:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            prediction = net(X_test_image)
        if isinstance(prediction, tuple or list):
            prediction = prediction[0]
        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        if not all_process:
            prediction += 1
        outputs[count_id: count_id + batch_size] = prediction
        count_id += batch_size
        print(count_id, outputs.shape)
    if not all_process:  # 背景赋值0，其它+1
        for i in range(height * width):
            if patch_labels[i] == 0:
                outputs[i] = 0
    return outputs.reshape(height, width)


def predict_more(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', batch_size=100, lines_per_batch=10):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width))
    pre_time_total = 0.

    image_patches = []
    positions = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(0, height, lines_per_batch):
        # 确定当前批次的行数
        current_batch_lines = min(lines_per_batch, height - i)

        for j in range(width):
            for k in range(current_batch_lines):
                row = i + k
                if row < height:
                    # 提取图像块
                    image_patch = X[row:row + patch_size, j:j + patch_size, :]

                    # 调整图像块的形状
                    if dim == 3:
                        image_patch = image_patch.reshape(1, *image_patch.shape, 1)
                        processed_patch = image_patch.transpose(0, 4, 3, 1, 2)
                    else:  # dim == 2
                        image_patch = image_patch.reshape(1, *image_patch.shape)
                        processed_patch = image_patch.transpose(0, 3, 1, 2)

                    image_patches.append(processed_patch)
                    positions.append((row, j))

            # 当达到批量大小时，进行预测
            if len(image_patches) >= batch_size:
                X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

                start.record()
                with torch.no_grad():
                    predictions = net(X_test_image)
                end.record()
                end.synchronize()
                pre_time_total += (start.elapsed_time(end) / 1000)

                # 处理预测结果并映射回输出
                predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
                for idx, (i_pos, j_pos) in enumerate(positions):
                    outputs[i_pos][j_pos] = predictions[idx] + 1

                # 清空列表，准备下一批次
                image_patches = []
                positions = []

        # 打印进度信息
        print(f"... ... rows {i} to {i + current_batch_lines - 1} handling ... ...")
        print('Time elapsed:', pre_time_total)

    # 处理剩余的图像块
    if len(image_patches) > 0:
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

        start.record()
        with torch.no_grad():
            predictions = net(X_test_image)
        end.record()
        end.synchronize()
        pre_time_total += (start.elapsed_time(end) / 1000)

        predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        for idx, (i_pos, j_pos) in enumerate(positions):
            outputs[i_pos][j_pos] = predictions[idx] + 1

    print("Total time:", pre_time_total)
    return outputs.astype(np.uint8)


def predict1_more(X, y, patch_size: int, net: nn.Module, dim=2, device='cuda', batch_size=50, lines_per_batch=100):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width))
    pre_time_total = 0.

    image_patches = []
    positions = []

    # CUDA事件用于计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(0, height, lines_per_batch):
        # 确定当前批次的行数
        current_batch_lines = min(lines_per_batch, height - i)

        for j in range(width):
            for k in range(current_batch_lines):
                row = i + k
                if row < height:
                    # 提取图像块
                    image_patch = X[row:row + patch_size, j:j + patch_size, :]

                    # 调整图像块的形状
                    if dim == 3:
                        image_patch = image_patch.reshape(1, *image_patch.shape, 1)
                        processed_patch = image_patch.transpose(0, 4, 3, 1, 2)
                    else:  # dim == 2
                        image_patch = image_patch.reshape(1, *image_patch.shape)
                        processed_patch = image_patch.transpose(0, 3, 1, 2)

                    image_patches.append(processed_patch)
                    positions.append((row, j))

            # 达到批量大小时进行批量预测
            if len(image_patches) >= batch_size:
                X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device, non_blocking=True)

                start.record()
                with torch.no_grad():
                    predictions = net(X_test_image)
                end.record()
                torch.cuda.synchronize()  # 等待GPU完成以获取精确时间
                pre_time_total += (start.elapsed_time(end) / 1000)

                # 处理预测结果并映射回输出
                predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
                for idx, (i_pos, j_pos) in enumerate(positions):
                    outputs[i_pos][j_pos] = predictions[idx] + 1

                # 清空列表，准备下一批次
                image_patches = []
                positions = []
        # 打印进度信息
        print(f"... ... rows {i} to {i + current_batch_lines - 1} handling ... ...")
        print('Time elapsed:', pre_time_total)

    # 处理剩余的图像块
    if len(image_patches) > 0:
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device, non_blocking=True)

        start.record()
        with torch.no_grad():
            predictions = net(X_test_image)
        end.record()
        torch.cuda.synchronize()
        pre_time_total += (start.elapsed_time(end) / 1000)

        predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        for idx, (i_pos, j_pos) in enumerate(positions):
            outputs[i_pos][j_pos] = predictions[idx] + 1

    print("Total time:", pre_time_total)
    return outputs.astype(np.uint8)

def predictv2(X, y, patch_size: int, net: nn.Module, dim=2, device='cuda', batch_size=256):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height, width = y.shape[:2]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width), dtype=np.uint8)
    pre_time_total = 0.0

    image_patches = []
    positions = []

    # 使用 cuda events 来计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 首先在所有图像块中进行批处理
    for i in range(height):
        for j in range(width):
            # 提取图像块并调整形状
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            if dim == 3:
                image_patch = image_patch[None, ..., None].transpose(0, 4, 3, 1, 2)
            else:
                image_patch = image_patch[None].transpose(0, 3, 1, 2)

            image_patches.append(image_patch)
            positions.append((i, j))

            # 达到批量大小后进行推理
            if len(image_patches) == batch_size:
                # 将批量数据转为张量并发送到 GPU
                X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)
                image_patches.clear()

                # 记录时间
                start.record()
                with torch.no_grad():
                    predictions = net(X_test_image)
                end.record()
                torch.cuda.synchronize()
                pre_time_total += start.elapsed_time(end) / 1000  # 转换为秒

                # 将结果映射到输出
                predictions = predictions.argmax(dim=1).cpu().numpy()
                for idx, (i_pos, j_pos) in enumerate(positions):
                    outputs[i_pos, j_pos] = predictions[idx] + 1
                positions.clear()  # 清理位置列表

        # 每10行打印一次进度
        if i % 10 == 0:
            print(f'... row {i} processed ... Time elapsed: {pre_time_total:.2f}s')

    # 处理剩余的图像块
    if image_patches:
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)
        start.record()
        with torch.no_grad():
            predictions = net(X_test_image)
        end.record()
        torch.cuda.synchronize()
        pre_time_total += start.elapsed_time(end) / 1000

        predictions = predictions.argmax(dim=1).cpu().numpy()
        for idx, (i_pos, j_pos) in enumerate(positions):
            outputs[i_pos, j_pos] = predictions[idx] + 1

    print("Total inference time:", pre_time_total)
    return outputs


def predictv3(X, y, patch_size: int, net: nn.Module, dim=2, device='cuda', batch_size=256):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height, width = y.shape[:2]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width), dtype=np.uint8)
    pre_time_total = 0.0

    # 准备存储图像块和位置的列表
    image_patches = []
    positions = []

    # 提前提取所有图像块，并将其直接转为 GPU 张量
    for i in range(height):
        for j in range(width):
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            if dim == 3:
                image_patch = image_patch[None, ..., None].transpose(0, 4, 3, 1, 2)
            else:
                image_patch = image_patch[None].transpose(0, 3, 1, 2)

            image_patches.append(image_patch)
            positions.append((i, j))

    # 将所有图像块转换为一个大张量，并提前加载到 GPU
    X_all_patches = torch.FloatTensor(np.vstack(image_patches)).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 按批次在 GPU 上运行推理
    for i in range(0, len(image_patches), batch_size):
        batch_patches = X_all_patches[i:i + batch_size]

        # 记录时间
        start.record()
        with torch.no_grad():
            predictions = net(batch_patches)
        end.record()
        torch.cuda.synchronize()
        pre_time_total += start.elapsed_time(end) / 1000  # 转换为秒

        # 取出批次对应的预测结果
        predictions = predictions.argmax(dim=1).cpu().numpy()
        batch_positions = positions[i:i + batch_size]
        for idx, (i_pos, j_pos) in enumerate(batch_positions):
            outputs[i_pos, j_pos] = predictions[idx] + 1

        # 每10个批次打印一次进度
        if i % (10 * batch_size) == 0:
            print(f'... processed up to patch {i} ... Time elapsed: {pre_time_total:.2f}s')

    print("Total inference time:", pre_time_total)
    return outputs


def predictv4(X, y, patch_size: int, net: nn.Module, dim=2, device='cuda', batch_size=100, row_batch_size=4):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height, width = y.shape[:2]
    X = zeroPadding(X, patch_size // 2)
    outputs = np.zeros((height, width), dtype=np.uint8)
    pre_time_total = 0.0

    # 使用 cuda events 来计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 按指定行数加载数据到 GPU
    for row_start in range(0, height, row_batch_size):
        row_end = min(row_start + row_batch_size, height)
        image_patches = []
        positions = []

        # 提取这些行的所有图像块
        for i in range(row_start, row_end):
            for j in range(width):
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                if dim == 3:
                    image_patch = image_patch[None, ..., None].transpose(0, 4, 3, 1, 2)
                else:
                    image_patch = image_patch[None].transpose(0, 3, 1, 2)

                image_patches.append(image_patch)
                positions.append((i, j))

        # 将提取的图像块转换为张量并传入 GPU
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

        # 批次处理 GPU 中的图像块
        for batch_start in range(0, X_test_image.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, X_test_image.shape[0])
            batch_patches = X_test_image[batch_start:batch_end]

            # 记录时间
            start.record()
            with torch.no_grad():
                predictions = net(batch_patches)
            end.record()
            torch.cuda.synchronize()
            pre_time_total += start.elapsed_time(end) / 1000  # 转换为秒

            # 处理预测结果
            batch_predictions = predictions.argmax(dim=1).cpu().numpy()
            batch_positions = positions[batch_start:batch_end]
            for idx, (i_pos, j_pos) in enumerate(batch_positions):
                outputs[i_pos, j_pos] = batch_predictions[idx] + 1

        print(f'Processed rows {row_start} to {row_end - 1}, Time elapsed: {pre_time_total:.2f}s')

    print("Total inference time:", pre_time_total)
    return outputs


def predictv5(X, y, patch_size: int, net: nn.Module, height, width, dim=2, device='cuda', batch_size=100, row_batch_size=4):
    assert dim in (2, 3)
    outputs = np.zeros((height, width), dtype=np.uint8)
    pre_time_total = 0.0

    # 使用 cuda events 来计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 按指定行数加载数据到 GPU
    for row_start in range(0, height, row_batch_size):
        row_end = min(row_start + row_batch_size, height)
        image_patches = []
        positions = []

        # 提取这些行的所有图像块
        for i in range(row_start, row_end):
            for j in range(width):
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                if dim == 3:
                    image_patch = image_patch[None, ..., None].transpose(0, 4, 3, 1, 2)
                else:
                    image_patch = image_patch[None].transpose(0, 3, 1, 2)

                image_patches.append(image_patch)
                positions.append((i, j))

        # 将提取的图像块转换为张量并传入 GPU
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

        # 批次处理 GPU 中的图像块
        for batch_start in range(0, X_test_image.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, X_test_image.shape[0])
            batch_patches = X_test_image[batch_start:batch_end]

            # 记录时间
            start.record()
            with torch.no_grad():
                predictions = net(batch_patches)
            end.record()
            torch.cuda.synchronize()
            pre_time_total += start.elapsed_time(end) / 1000  # 转换为秒

            # 处理预测结果
            batch_predictions = predictions.argmax(dim=1).cpu().numpy()
            batch_positions = positions[batch_start:batch_end]
            for idx, (i_pos, j_pos) in enumerate(batch_positions):
                outputs[i_pos, j_pos] = batch_predictions[idx] + 1

            # 清理中间变量
            del batch_patches, predictions, batch_predictions, batch_positions

        # 每个 row_batch 结束后清理
        del X_test_image, image_patches, positions

        if device == 'cuda':
            torch.cuda.empty_cache()

        print(f'Processed rows {row_start} to {row_end - 1}, Time elapsed: {pre_time_total:.2f}s')

    print("Total inference time:", pre_time_total)
    return outputs,pre_time_total



def predictv7(X, y, patch_size: int, net: nn.Module, height, width, dim=2, device='cuda', batch_size=100, row_batch_size=4):
    assert dim in (2, 3)
    class_num=len(np.unique(y))-1
    outputs = np.zeros((height, width), dtype=np.uint8)
    features = np.zeros((height, width, class_num), dtype=np.float32)
    pre_time_total = 0.0

    # 使用 cuda events 来计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 按指定行数加载数据到 GPU
    for row_start in range(0, height, row_batch_size):
        row_end = min(row_start + row_batch_size, height)
        image_patches = []
        positions = []

        # 提取这些行的所有图像块
        for i in range(row_start, row_end):
            for j in range(width):
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                if dim == 3:
                    image_patch = image_patch[None, ..., None].transpose(0, 4, 3, 1, 2)
                else:
                    image_patch = image_patch[None].transpose(0, 3, 1, 2)

                image_patches.append(image_patch)
                positions.append((i, j))

        # 将提取的图像块转换为张量并传入 GPU
        X_test_image = torch.FloatTensor(np.vstack(image_patches)).to(device)

        # 批次处理 GPU 中的图像块
        for batch_start in range(0, X_test_image.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, X_test_image.shape[0])
            batch_patches = X_test_image[batch_start:batch_end]

            # 记录时间
            start.record()
            with torch.no_grad():
                predictions, emb_fea = net(batch_patches)
            end.record()
            torch.cuda.synchronize()
            pre_time_total += start.elapsed_time(end) / 1000  # 转换为秒

            # 处理预测结果
            batch_predictions = predictions.argmax(dim=1).cpu().numpy()
            batch_positions = positions[batch_start:batch_end]
            for idx, (i_pos, j_pos) in enumerate(batch_positions):
                outputs[i_pos, j_pos] = batch_predictions[idx] + 1
                features[i_pos, j_pos, :] = emb_fea[idx].cpu().numpy()

            # 清理中间变量
            del batch_patches, predictions, batch_predictions, batch_positions

        # 每个 row_batch 结束后清理
        del X_test_image, image_patches, positions

        if device == 'cuda':
            torch.cuda.empty_cache()

        print(f'Processed rows {row_start} to {row_end - 1}, Time elapsed: {pre_time_total:.2f}s')

    print("Total inference time:", pre_time_total)
    return outputs,features,pre_time_total


