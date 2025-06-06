# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # --- 开始修改部分：实现双Y轴绘图 ---
    fig, ax1 = plt.subplots(figsize=(10, 6)) # 创建图形和左侧Y轴 (mIoU)
    ax2 = ax1.twinx() # 创建共享X轴的右侧Y轴 (Loss)

    # 设置X轴标签
    ax1.set_xlabel(args.xlabel if args.xlabel else 'Epoch')
    # 设置左Y轴标签 (mIoU)
    ax1.set_ylabel('mIoU', color='blue') # 可以给轴标签设置颜色
    ax1.tick_params(axis='y', labelcolor='blue') # Y轴刻度标签颜色
    
    # 设置右Y轴标签 (Loss)
    ax2.set_ylabel('Loss', color='red') # 可以给轴标签设置颜色
    ax2.tick_params(axis='y', labelcolor='red') # Y轴刻度标签颜色

    # 定义颜色映射，为每个模型（json_log）分配一个独特的颜色
    # 这确保了HDE-SegNet和LEFormer在mIoU和Loss曲线上使用同一颜色
    colors = plt.cm.get_cmap('tab10', len(args.json_logs))
    
    # 定义mIoU和Loss曲线的线型和标记，以便区分它们
    miou_marker = 'o'   # mIoU使用圆形标记
    loss_marker = 'x'   # Loss使用叉形标记
    miou_linestyle = '-'  # mIoU使用实线
    loss_linestyle = '--' # Loss使用虚线

    all_handles = [] # 用于收集所有图例句柄
    all_labels = []  # 用于收集所有图例标签

    # 遍历每个日志字典 (每个JSON日志文件代表一个模型，例如HDE-SegNet或LEFormer)
    for i, log_dict in enumerate(log_dicts):
        # 从日志文件路径中提取模型名称
        model_name = args.json_logs[i].split('/')[-1].replace('.json', '')
        current_model_color = colors(i) # 获取当前模型的专属颜色

        epochs = sorted(list(log_dict.keys()))

        # 数据收集：mIoU
        plot_epochs_miou = []
        plot_values_miou = []
        # 数据收集：Loss
        plot_epochs_loss = []
        plot_values_loss = []

        # 遍历每个epoch，收集mIoU和Loss数据
        for epoch in epochs:
            epoch_logs = log_dict[epoch]

            # 处理 mIoU 数据
            if 'mIoU' in epoch_logs and epoch_logs['mIoU']:
                plot_epochs_miou.append(epoch)
                plot_values_miou.append(epoch_logs['mIoU'][0]) # 假设mIoU每个epoch只有一个值

            # 处理 Loss 数据 (计算训练模式下的平均Loss)
            if 'loss' in epoch_logs and 'mode' in epoch_logs:
                epoch_train_losses = []
                # 遍历当前epoch的所有日志条目
                for idx in range(len(epoch_logs['mode'])):
                    # 确保索引有效，且是训练模式的loss
                    if epoch_logs['mode'][idx] == 'train' and idx < len(epoch_logs['loss']):
                        epoch_train_losses.append(epoch_logs['loss'][idx])

                if epoch_train_losses:
                    plot_epochs_loss.append(epoch)
                    plot_values_loss.append(np.mean(epoch_train_losses))
                else:
                    # 如果当前epoch没有训练Loss数据，则添加NaN以保持曲线的连续性（或显示断点）
                    plot_epochs_loss.append(epoch)
                    plot_values_loss.append(float('nan'))

        # 绘制当前模型的 mIoU 曲线 (在左Y轴)
        if plot_epochs_miou: # 只有当有数据时才绘制
            line_miou, = ax1.plot(plot_epochs_miou, plot_values_miou,
                                  label=f'{model_name} mIoU', # 图例标签
                                  color=current_model_color, # 使用模型的专属颜色
                                  marker=miou_marker,
                                  linestyle=miou_linestyle)
            all_handles.append(line_miou)
            all_labels.append(f'{model_name} mIoU')
        
        # 绘制当前模型的 Loss 曲线 (在右Y轴)
        if plot_epochs_loss: # 只有当有数据时才绘制
            line_loss, = ax2.plot(plot_epochs_loss, plot_values_loss,
                                  label=f'{model_name} Loss', # 图例标签
                                  color=current_model_color, # 使用模型的专属颜色
                                  marker=loss_marker,
                                  linestyle=loss_linestyle)
            all_handles.append(line_loss)
            all_labels.append(f'{model_name} Loss')

    # 设置X轴的刻度为整数 (epochs)
    # 收集所有模型的epoch范围以确保X轴刻度一致
    all_epochs_overall = set()
    for log_dict in log_dicts:
        all_epochs_overall.update(log_dict.keys())

    if all_epochs_overall:
        max_e = max(all_epochs_overall)
        ax1.set_xticks(range(0, max_e + 1))  # 从0开始
        ax1.set_xlim(-0.5, max_e + 0.5)      # 同样从-0.5开始，留一点边距

    # 将所有曲线的图例合并到一个图例中，并放置在图的右侧外部
    # loc='upper left' 是相对于 bbox_to_anchor 指定的坐标点
    # bbox_to_anchor=(1.05, 1) 表示放置在(1.05, 1)这个坐标点，(1,1)是右上角
    # borderaxespad=0. 表示图例与锚点之间的距离为0
    fig.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # 设置整个图表的标题 (使用fig.suptitle以免与轴标题冲突)
    if args.title is not None:
        fig.suptitle(args.title)

    # 调整布局，为图例留出空间，rect参数定义了子图的边界 [left, bottom, right, top]
    # 这里将子图的右边界设置为0.85，为右侧的图例腾出15%的空间
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 

    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
    
    
    # 关闭图形以释放内存
    plt.close(fig) 
    # --- 结束修改部分 ---


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'], # 默认值，但当需要双Y轴时，用户应指定 --keys mIoU loss
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None, # 图例现在由代码自动生成，更清晰
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument(
        '--xlabel',
        type=str,
        default=None,
        help='Custom label for the x-axis. If not provided, defaults to "epoch" or "iter".')
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without epoch field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)

    # 打印表格数据
    print_metrics_table(log_dicts, args)

    # 绘图
    plot_curve(log_dicts, args)



def print_metrics_table(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        model_name = args.json_logs[i].split('/')[-1]
        print(f"\n=== {model_name} 数据 ===")
        print(f"{'Epoch':<8}{'mIoU':<10}{'Loss':<10}")
        
        epochs = sorted(log_dict.keys())
        for epoch in epochs:
            logs = log_dict[epoch]
            # 取第一个 mIoU（通常每个 epoch 只有一个）
            miou = logs.get('mIoU', [None])[0]
            # 提取训练 Loss（平均值）
            losses = [logs['loss'][i] for i in range(len(logs.get('loss', [])))
                      if 'mode' in logs and logs['mode'][i] == 'train']
            loss_avg = np.mean(losses) if losses else None
            # 打印
            if miou is not None and loss_avg is not None:
                print(f"{epoch:<8}{miou:<10.4f}{loss_avg:<10.4f}")
            elif miou is not None:
                print(f"{epoch:<8}{miou:<10.4f}{'N/A':<10}")
            elif loss_avg is not None:
                print(f"{epoch:<8}{'N/A':<10}{loss_avg:<10.4f}")


if __name__ == '__main__':
    main()

