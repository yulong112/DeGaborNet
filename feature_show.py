import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# from collections import defaultdict
from skimage import exposure
from scipy.linalg import pinv


# ==================== 子函数1: 可视化特征图（降维到3通道显示） ====================
def visualize_feature_maps(emb_features, y, save_path=None, method='pca'):
    """
    将64通道特征图降维到3通道显示为彩色图像

    Args:
        emb_features: np.ndarray, shape [H, W, C]，整个图像的特征立方体
        y: np.ndarray, shape [H, W]，标签图
        save_path: str, 保存路径，None则显示
        method: str, 降维方法 'pca' 或 'mean'（均值降维）
    """
    H, W, C = emb_features.shape

    # 检查输入
    if emb_features.shape[:2] != y.shape:
        raise ValueError(f"Feature shape {emb_features.shape[:2]} != Label shape {y.shape}")

    # 将特征图降维到3通道
    if method == 'pca':
        # 方法1: PCA降维到3通道
        features_flat = emb_features.reshape(-1, C)  # [H*W, C]

        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_flat)

        # PCA降维到3个主成分
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_scaled)  # [H*W, 3]

        # 归一化到[0,1]范围用于显示
        features_pca = (features_pca - features_pca.min(axis=0)) / (
                    features_pca.max(axis=0) - features_pca.min(axis=0))

        # 重塑回图像格式
        rgb_image = features_pca.reshape(H, W, 3)

        # 归一化到0-255
        # rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-10)
        # rgb_image = (rgb_image * 255).astype(np.uint8)

        # 直方图均衡化增强对比度
        rgb_image_equalized = np.zeros_like(rgb_image)
        for i in range(3):
            channel = rgb_image[:, :, i]
            # 标准化到0-255
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)
            channel_norm = (channel_norm * 255).astype(np.uint8)
            # 直方图均衡化
            rgb_image_equalized[:, :, i] = exposure.equalize_hist(channel_norm)

        rgb_image = rgb_image_equalized.copy()
        title = f'PCA Features (explained var: {pca.explained_variance_ratio_.sum():.2%})'

    elif method == 'mean':
        # 方法2: 将64通道分成3组，每组取均值
        split_size = C // 3
        rgb_image = np.zeros((H, W, 3))
        for i in range(3):
            start = i * split_size
            end = (i + 1) * split_size if i < 2 else C
            rgb_image[:, :, i] = np.mean(emb_features[:, :, start:end], axis=2)

        # 归一化到0-255
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-10)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        title = 'Channel Group Mean (RGB)'

    elif method == 'tsne':
        # 方法3: t-SNE降维（计算较慢，适合小图像）
        try:
            from sklearn.manifold import TSNE
            features_flat = emb_features.reshape(-1, C)
            tsne = TSNE(n_components=3, perplexity=30, random_state=42)
            features_tsne = tsne.fit_transform(features_flat)
            rgb_image = features_tsne.reshape(H, W, 3)
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-10)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            title = 't-SNE Features'
        except ImportError:
            print("t-SNE not available, using PCA instead")
            return visualize_feature_maps(emb_features, y, save_path, method='pca')
    else:
        raise ValueError(f"Unknown method: {method}")

    # # 创建可视化
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #
    # # 图1: 彩色特征图
    # ax1 = axes[0]
    # ax1.imshow(rgb_image)
    # # ax1.set_title(title)
    # ax1.axis('off')
    #
    # # 图2: 标签图
    # ax2 = axes[1]
    # # 为标签图创建彩色映射
    # unique_labels = np.unique(y)
    # num_labels = len(unique_labels)
    # if num_labels <= 10:
    #     cmap = plt.cm.tab10
    # else:
    #     cmap = plt.cm.tab20
    #
    # im = ax2.imshow(y, cmap=cmap, interpolation='nearest')
    # ax2.set_title(f'Ground Truth Labels\n(classes: {unique_labels.tolist()})')
    # ax2.axis('off')
    # plt.colorbar(im, ax=ax2, ticks=unique_labels)
    #
    # # 图3: 特征图 + 标签叠加
    # ax3 = axes[2]
    # ax3.imshow(rgb_image, alpha=0.7)
    # # 绘制标签边界（可选）
    # from scipy import ndimage
    # # 检测标签边界
    # if len(unique_labels) > 1:
    #     edges = np.zeros_like(y, dtype=bool)
    #     for label in unique_labels:
    #         if label == 0:  # 跳过背景
    #             continue
    #         mask = (y == label).astype(np.uint8)
    #         # 使用sobel边缘检测
    #         edges_mask = ndimage.sobel(mask, axis=0) ** 2 + ndimage.sobel(mask, axis=1) ** 2
    #         edges |= edges_mask > 0.1
    #     # 在特征图上绘制边界
    #     y_edges = np.zeros_like(y, dtype=float)
    #     y_edges[edges] = 1
    #     ax3.imshow(y_edges, cmap='Reds', alpha=0.5, interpolation='nearest')
    #
    # ax3.set_title('Feature Map with Label Boundaries')
    # ax3.axis('off')

    fig, ax = plt.subplots()

    # 图1: 彩色特征图
    ax.imshow(rgb_image)
    ax.axis('off')

    # plt.tight_layout(pad=0)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
        # plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()

    return rgb_image


# ==================== 子函数2: 计算Fisher Ratio ====================
def compute_fisher_ratio(emb_features, y, ignore_label=[0]):
    """
    计算特征的Fisher Ratio（可分性指标）
    公式: FR = (类间方差) / (类内方差)
    值越大表示特征区分度越高

    Args:
        emb_features: np.ndarray, shape [H, W, C]，整个图像的特征立方体
        y: np.ndarray, shape [H, W]，标签图
        ignore_label: int, 忽略的标签（通常是背景）

    Returns:
        dict: 包含各类别FR值及平均值
    """
    H, W, C = emb_features.shape

    # 检查输入
    if emb_features.shape[:2] != y.shape:
        raise ValueError(f"Feature shape {emb_features.shape[:2]} != Label shape {y.shape}")

    # 展平空间维度
    features_flat = emb_features.reshape(-1, C)  # [H*W, C]
    labels_flat = y.flatten()  # [H*W]

    # 过滤掉忽略标签
    # valid_mask = labels_flat != ignore_label
    valid_mask = ~np.isin(labels_flat, ignore_label)
    if not valid_mask.any():
        raise ValueError("No valid pixels found!")

    features_valid = features_flat[valid_mask]  # [N_valid, C]
    labels_valid = labels_flat[valid_mask]  # [N_valid]

    # 获取所有类别
    unique_labels = np.unique(labels_valid)
    num_classes = len(unique_labels)
    print(f"Found {num_classes} classes: {unique_labels}")

    # 统计信息
    class_means = {}
    class_counts = {}
    class_covs = {}

    overall_mean = np.mean(features_valid, axis=0, keepdims=True)  # [1, C]

    for label in unique_labels:
        mask = labels_valid == label
        class_feat = features_valid[mask]  # [N_class, C]

        class_means[label] = np.mean(class_feat, axis=0, keepdims=True)  # [1, C]
        class_counts[label] = class_feat.shape[0]

        # 类内协方差矩阵
        if class_feat.shape[0] > 1:
            class_covs[label] = np.cov(class_feat, rowvar=False)  # [C, C]
        else:
            class_covs[label] = np.zeros((C, C))

    # 计算类间散度矩阵 (Between-class scatter matrix)
    Sb = np.zeros((C, C))
    for label in unique_labels:
        mean_diff = class_means[label] - overall_mean  # [1, C]
        Sb += class_counts[label] * np.dot(mean_diff.T, mean_diff)
    Sb = Sb / num_classes  # [C, C]

    # 计算类内散度矩阵 (Within-class scatter matrix)
    Sw = np.zeros((C, C))
    for label in unique_labels:
        Sw += class_covs[label] * class_counts[label]
    Sw = Sw / num_classes  # [C, C]

    # Fisher Ratio: 迹(Sb) / 迹(Sw)
    between_trace = np.trace(Sb)
    within_trace = np.trace(Sw)
    # fisher_ratio = between_trace / (within_trace + 1e-10)
    fisher_ratio = between_trace / (within_trace)

    # 计算每个类别的单独FR（该类与其他类的可分性）
    per_class_fr = {}
    for label in unique_labels:
        # 该类均值
        class_mean = class_means[label]  # [1, C]

        # 其他类的均值
        other_labels = [l for l in unique_labels if l != label]
        if other_labels:
            other_means = np.vstack([class_means[l] for l in other_labels])  # [num_other, C]
            other_mean = np.mean(other_means, axis=0, keepdims=True)  # [1, C]

            # 类间差异
            mean_diff = class_mean - other_mean  # [1, C]
            between_class = np.sum(mean_diff ** 2)
        else:
            between_class = 0

        # 类内方差（迹）
        within_class = np.trace(class_covs[label]) if np.trace(class_covs[label]) > 0 else 1e-10

        per_class_fr[label] = between_class / within_class

    results = {
        'fisher_ratio': fisher_ratio,
        'per_class_fisher_ratio': per_class_fr,
        'class_counts': class_counts,
        'num_classes': num_classes,
        'total_pixels': np.sum(list(class_counts.values())),
        'between_class_trace': between_trace,
        'within_class_trace': within_trace,
        'class_means': class_means,
        'class_covariances': class_covs
    }

    return results



def calculate_fisher_score(feature_data, gt, exclude_classes=[0]):
    """
    计算特征的Fisher判别分数

    参数:
    feature_data: 3D特征矩阵 [H, W, C] 或 2D特征矩阵 [N, C]
    gt: 对应的ground truth标签 [H, W] 或 [N]

    返回:
    fisher_score: Fisher判别分数
    between_class_scatter: 类间散度
    within_class_scatter: 类内散度
    """

    # 1. 将3D特征矩阵展平为2D
    if feature_data.ndim == 3:
        H, W, C = feature_data.shape
        feature_2d = feature_data.reshape(-1, C)  # [H*W, C]
        gt_flat = gt.reshape(-1)  # [H*W]
    else:
        feature_2d = feature_data
        gt_flat = gt

    # 2. 去除gt中为0的部分（假设0表示背景或无效区域）
    valid_mask = gt_flat != 0
    # valid_mask = ~np.isin(gt_flat, exclude_classes)
    features_valid = feature_2d[valid_mask]  # [N_valid, C]
    labels_valid = gt_flat[valid_mask]  # [N_valid]

    if len(features_valid) == 0:
        raise ValueError("没有有效的标签数据")

    # 3. 获取唯一的类别标签
    unique_labels = np.unique(labels_valid)
    n_classes = len(unique_labels)
    n_features = features_valid.shape[1]

    print(f"有效样本数: {len(features_valid)}")
    print(f"类别数: {n_classes}")
    print(f"特征维度: {n_features}")
    print(f"类别标签: {unique_labels}")

    # 4. 计算全局均值
    global_mean = np.mean(features_valid, axis=0)  # [C]

    # 5. 计算类内散度矩阵(S_w)和类间散度矩阵(S_b)
    S_w = np.zeros((n_features, n_features))  # 类内散度矩阵
    S_b = np.zeros((n_features, n_features))  # 类间散度矩阵

    for label in unique_labels:
        # 获取当前类的特征
        class_features = features_valid[labels_valid == label]  # [N_class, C]
        n_class = len(class_features)

        # 确保是2D数组
        if class_features.ndim == 1:
            class_features = class_features.reshape(1, -1)

        # 计算当前类的均值
        class_mean = np.mean(class_features, axis=0)  # [C]

        # 计算类内散度
        # class_scatter = np.zeros((n_features, n_features))
        # for feature_vec in class_features:
        #     diff = (feature_vec - class_mean).reshape(-1, 1)  # [C, 1]
        #     class_scatter += diff @ diff.T  # [C, C]

        # 向量化计算类内散度矩阵
        centered = class_features - class_mean  # [N_class, C]
        class_scatter = centered.T @ centered  # [C, C]

        S_w += class_scatter

        # 计算类间散度
        mean_diff = (class_mean - global_mean).reshape(-1, 1)  # [C, 1]
        S_b += n_class * (mean_diff @ mean_diff.T)  # [C, C]

    # 6. 计算Fisher判别分数
    # 方法1: 基于迹的Fisher分数 (更稳定)
    fisher_trace = np.trace(S_b) / np.trace(S_w) if np.trace(S_w) > 0 else 0

    # 方法2: 基于广义特征值的Fisher分数 (理论上更准确，但可能不稳定)
    try:
        eigenvalues = np.linalg.eigvals(pinv(S_w) @ S_b)
        fisher_eigen = np.mean(eigenvalues[eigenvalues > 0]) if np.any(eigenvalues > 0) else 0
    except:
        fisher_eigen = fisher_trace

    return {
        'fisher_score_trace': fisher_trace,
        'fisher_score_eigen': fisher_eigen,
        'between_class_scatter': S_b,
        'within_class_scatter': S_w,
        'n_classes': n_classes,
        'n_samples': len(features_valid),
        'class_distribution': [np.sum(labels_valid == label) for label in unique_labels]
    }



# ==================== 子函数3: 可视化特征分布（散点图） ====================
def visualize_feature_distribution(emb_features, y, save_path=None, ignore_label=255):
    """
    使用PCA降维到2D可视化特征分布

    Args:
        emb_features: np.ndarray, shape [H, W, C]
        y: np.ndarray, shape [H, W]
        save_path: str, 保存路径
        ignore_label: int, 忽略的标签
    """
    H, W, C = emb_features.shape

    # 展平
    features_flat = emb_features.reshape(-1, C)
    labels_flat = y.flatten()

    # 过滤背景
    valid_mask = labels_flat != ignore_label
    features_valid = features_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]

    # 随机采样以减少计算量（如果像素太多）
    max_samples = 10000
    if len(features_valid) > max_samples:
        indices = np.random.choice(len(features_valid), max_samples, replace=False)
        features_valid = features_valid[indices]
        labels_valid = labels_valid[indices]

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)

    # PCA降维到2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: PCA散点图
    ax1 = axes[0]
    unique_labels = np.unique(labels_valid)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels_valid == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=f'Class {label}', s=10, alpha=0.6)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('Feature Distribution (PCA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 各类别特征均值的热力图
    ax2 = axes[1]
    # 计算每个类别的特征均值（只显示前16通道）
    num_channels = min(16, C)
    class_means_matrix = []
    class_names = []

    for label in unique_labels:
        mask = labels_valid == label
        class_mean = np.mean(features_valid[mask, :num_channels], axis=0)
        class_means_matrix.append(class_mean)
        class_names.append(f'Class {label}')

    class_means_matrix = np.array(class_means_matrix)

    im = ax2.imshow(class_means_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Feature Channel')
    ax2.set_ylabel('Class')
    ax2.set_title(f'Class-wise Feature Means (first {num_channels} channels)')
    ax2.set_xticks(range(num_channels))
    ax2.set_xticklabels([f'C{i}' for i in range(num_channels)], rotation=45)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ==================== 子函数4: 可视化Fisher Ratio结果 ====================
def visualize_fisher_ratio(fisher_results, save_path=None):
    """
    可视化Fisher Ratio分析结果

    Args:
        fisher_results: dict, compute_fisher_ratio返回的结果
        save_path: str, 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 获取数据
    classes = list(fisher_results['per_class_fisher_ratio'].keys())
    fr_values = [fisher_results['per_class_fisher_ratio'][c] for c in classes]
    counts = [fisher_results['class_counts'][c] for c in classes]

    # 图1: 各类别的Fisher Ratio
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(np.array(fr_values) / max(fr_values + [1]))
    bars = ax1.bar(range(len(classes)), fr_values, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([f'Class {c}' for c in classes])
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Fisher Ratio')
    ax1.set_title(f'Per-Class Fisher Ratio\nOverall FR: {fisher_results["fisher_ratio"]:.4f}')
    ax1.grid(True, alpha=0.3)

    # 在柱状图上显示数值
    for bar, val in zip(bars, fr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 图2: 类别分布和FR对比
    ax2 = axes[1]
    ax2_bar = ax2
    bars2 = ax2_bar.bar(range(len(classes)), counts, color='lightblue', alpha=0.7)
    ax2_bar.set_xlabel('Class')
    ax2_bar.set_ylabel('Number of Pixels', color='blue')
    ax2_bar.tick_params(axis='y', labelcolor='blue')
    ax2_bar.set_xticks(range(len(classes)))
    ax2_bar.set_xticklabels([f'Class {c}' for c in classes])

    ax2_line = ax2.twinx()
    ax2_line.plot(range(len(classes)), fr_values, 'ro-', linewidth=2,
                  markersize=8, label='Fisher Ratio')
    ax2_line.set_ylabel('Fisher Ratio', color='red')
    ax2_line.tick_params(axis='y', labelcolor='red')

    ax2.set_title('Class Distribution vs Fisher Ratio')
    ax2.grid(True, alpha=0.3)

    # 图3: 类间vs类内方差
    ax3 = axes[2]
    between = fisher_results['between_class_trace']
    within = fisher_results['within_class_trace']

    ax3.bar(['Between-Class', 'Within-Class'],
            [between, within],
            color=['green', 'orange'],
            alpha=0.7)
    ax3.set_ylabel('Trace of Scatter Matrix')
    ax3.set_title(
        f'Scatter Matrix Comparison\nFR = {between:.2f} / {within:.2f} = {fisher_results["fisher_ratio"]:.4f}')
    ax3.grid(True, alpha=0.3)

    # 显示数值
    ax3.text(0, between / 2, f'{between:.2f}', ha='center', va='center', fontsize=12)
    ax3.text(1, within / 2, f'{within:.2f}', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fisher Ratio visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()

