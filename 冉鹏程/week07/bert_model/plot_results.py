import json
from pathlib import Path
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "hyperparam_tuning"


def load_results(results_file: Path) -> List[Dict]:
    """加载训练结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(results: List[Dict], output_dir: Path):
    """绘制所有运行的训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for run in results:
        epochs = [e['epoch'] for e in run['epochs']]
        train_loss = [e['train_loss'] for e in run['epochs']]
        val_loss = [e['val_loss'] for e in run['epochs']]
        val_f1 = [e['val_entity_f1'] for e in run['epochs']]

        config_label = f"Run{run['run_id']}_lr{run['config']['lr']}_bs{run['config']['batch_size']}_do{run['config']['dropout']}"

        axes[0, 0].plot(epochs, train_loss, label=config_label, marker='o', markersize=4)
        axes[0, 1].plot(epochs, val_loss, label=config_label, marker='o', markersize=4)
        axes[1, 0].plot(epochs, val_f1, label=config_label, marker='o', markersize=4)

    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Validation F1 Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线图已保存: {output_dir / 'training_curves.png'}")


def plot_best_runs(results: List[Dict], output_dir: Path, top_k: int = 5):
    """绘制表现最好的k个运行的曲线"""
    sorted_results = sorted(results, key=lambda x: x['best_val_f1'], reverse=True)
    top_results = sorted_results[:top_k]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for run in top_results:
        epochs = [e['epoch'] for e in run['epochs']]
        val_loss = [e['val_loss'] for e in run['epochs']]
        val_f1 = [e['val_entity_f1'] for e in run['epochs']]

        config_label = f"Run{run['run_id']}_lr{run['config']['lr']}_bs{run['config']['batch_size']}_do{run['config']['dropout']}_F1{run['best_val_f1']:.4f}"

        axes[0].plot(epochs, val_loss, label=config_label, marker='o', markersize=6, linewidth=2)
        axes[1].plot(epochs, val_f1, label=config_label, marker='o', markersize=6, linewidth=2)

    axes[0].set_title(f'Top {top_k} Runs - Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].set_title(f'Top {top_k} Runs - Validation F1', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_runs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Top {top_k} 曲线图已保存: {output_dir / 'top_runs.png'}")


def plot_hyperparameter_analysis(results: List[Dict], output_dir: Path):
    """分析超参数对结果的影响"""
    lrs = []
    batch_sizes = []
    dropouts = []
    head_lr_mults = []
    f1_scores = []

    for run in results:
        lrs.append(run['config']['lr'])
        batch_sizes.append(run['config']['batch_size'])
        dropouts.append(run['config']['dropout'])
        head_lr_mults.append(run['config']['head_lr_mult'])
        f1_scores.append(run['best_val_f1'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scatter1 = axes[0, 0].scatter(lrs, f1_scores, c=batch_sizes, s=100, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
    axes[0, 0].set_ylabel('Best Val F1', fontsize=12)
    axes[0, 0].set_title('LR vs F1 (colored by batch size)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Batch Size')

    scatter2 = axes[0, 1].scatter(dropouts, f1_scores, c=lrs, s=100, cmap='plasma', alpha=0.7)
    axes[0, 1].set_xlabel('Dropout', fontsize=12)
    axes[0, 1].set_ylabel('Best Val F1', fontsize=12)
    axes[0, 1].set_title('Dropout vs F1 (colored by LR)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Learning Rate')

    batch_unique = sorted(list(set(batch_sizes)))
    for bs in batch_unique:
        bs_f1s = [f for f, b in zip(f1_scores, batch_sizes) if b == bs]
        axes[1, 0].boxplot(bs_f1s, positions=[batch_unique.index(bs) + 1], widths=0.6,
                          labels=[str(bs)])
    axes[1, 0].set_xlabel('Batch Size', fontsize=12)
    axes[1, 0].set_ylabel('Best Val F1', fontsize=12)
    axes[1, 0].set_title('Batch Size Distribution', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    head_unique = sorted(list(set(head_lr_mults)))
    for hm in head_unique:
        hm_f1s = [f for f, h in zip(f1_scores, head_lr_mults) if h == hm]
        axes[1, 1].boxplot(hm_f1s, positions=[head_unique.index(hm) + 1], widths=0.6,
                          labels=[str(hm)])
    axes[1, 1].set_xlabel('Head LR Multiplier', fontsize=12)
    axes[1, 1].set_ylabel('Best Val F1', fontsize=12)
    axes[1, 1].set_title('Head LR Multiplier Distribution', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparam_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"超参数分析图已保存: {output_dir / 'hyperparam_analysis.png'}")


def plot_combined_analysis(results: List[Dict], output_dir: Path):
    """组合分析图 - 3D视角和热力图"""
    fig = plt.figure(figsize=(16, 6))

    lrs = np.array([r['config']['lr'] for r in results])
    batch_sizes = np.array([r['config']['batch_size'] for r in results])
    dropouts = np.array([r['config']['dropout'] for r in results])
    f1_scores = np.array([r['best_val_f1'] for r in results])

    ax1 = fig.add_subplot(121)
    lr_unique = sorted(list(set(lrs)))
    bs_unique = sorted(list(set(batch_sizes)))

    heatmap_data = np.zeros((len(lr_unique), len(bs_unique)))
    for i, lr in enumerate(lr_unique):
        for j, bs in enumerate(bs_unique):
            mask = (lrs == lr) & (batch_sizes == bs)
            if np.any(mask):
                heatmap_data[i, j] = np.mean(f1_scores[mask])

    im = ax1.imshow(heatmap_data, cmap='YlGnBu_r', aspect='auto')
    ax1.set_xticks(np.arange(len(bs_unique)))
    ax1.set_yticks(np.arange(len(lr_unique)))
    ax1.set_xticklabels([str(b) for b in bs_unique])
    ax1.set_yticklabels([f'{lr:.0e}' for lr in lr_unique])
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('F1 Score: LR vs Batch Size', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Mean F1 Score')

    for i in range(len(lr_unique)):
        for j in range(len(bs_unique)):
            if heatmap_data[i, j] > 0:
                ax1.text(j, i, f'{heatmap_data[i, j]:.3f}',
                        ha='center', va='center', color='black', fontsize=10)

    ax2 = fig.add_subplot(122)
    do_unique = sorted(list(set(dropouts)))
    heatmap_data2 = np.zeros((len(lr_unique), len(do_unique)))
    for i, lr in enumerate(lr_unique):
        for j, d in enumerate(do_unique):
            mask = (lrs == lr) & (dropouts == d)
            if np.any(mask):
                heatmap_data2[i, j] = np.mean(f1_scores[mask])

    im2 = ax2.imshow(heatmap_data2, cmap='YlGnBu_r', aspect='auto')
    ax2.set_xticks(np.arange(len(do_unique)))
    ax2.set_yticks(np.arange(len(lr_unique)))
    ax2.set_xticklabels([str(d) for d in do_unique])
    ax2.set_yticklabels([f'{lr:.0e}' for lr in lr_unique])
    ax2.set_xlabel('Dropout', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('F1 Score: LR vs Dropout', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Mean F1 Score')

    for i in range(len(lr_unique)):
        for j in range(len(do_unique)):
            if heatmap_data2[i, j] > 0:
                ax2.text(j, i, f'{heatmap_data2[i, j]:.3f}',
                        ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图分析已保存: {output_dir / 'heatmap_analysis.png'}")


def generate_summary_report(results: List[Dict], output_dir: Path):
    """生成总结报告"""
    sorted_results = sorted(results, key=lambda x: x['best_val_f1'], reverse=True)

    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("超参数调优总结报告\n")
        f.write("="*70 + "\n\n")

        f.write(f"总运行次数: {len(results)}\n\n")

        f.write("Top 10 配置:\n")
        f.write("-"*70 + "\n")
        for i, run in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. Best F1: {run['best_val_f1']:.4f}\n")
            f.write(f"   Config: {run['config']}\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("所有运行详情:\n")
        f.write("="*70 + "\n")
        for run in sorted_results:
            f.write(f"Run {run['run_id']:2d} | F1: {run['best_val_f1']:.4f} | {run['config']}\n")

    print(f"总结报告已保存: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="可视化超参数调优结果")
    parser.add_argument("--results_file", type=Path,
                        default=DEFAULT_OUTPUT_DIR / "all_results.json",
                        help="训练结果JSON文件路径")
    parser.add_argument("--output_dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR,
                        help="图表保存目录")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_file.exists():
        print(f"错误: 结果文件不存在: {args.results_file}")
        print(f"请先运行 hyperparameter_tuning.py 进行训练")
        return

    results = load_results(args.results_file)
    print(f"加载了 {len(results)} 次运行结果")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(results, args.output_dir)
    plot_best_runs(results, args.output_dir, top_k=min(5, len(results)))
    plot_hyperparameter_analysis(results, args.output_dir)
    plot_combined_analysis(results, args.output_dir)
    generate_summary_report(results, args.output_dir)

    print("\n所有分析图表生成完成！")


if __name__ == '__main__':
    main()
