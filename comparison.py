import matplotlib.pyplot as plt
import numpy as np


# ฟังก์ชันเปรียบเทียบประสิทธิภาพของทั้ง 3 โมเดล (SVM ปกติ, Bagging, Boosting) ทั้งความแม่นยำ และ เวลา (Train / Test)
def compare_models_accuracy_and_time(
    svm_train_accuracy,
    svm_test_accuracy,
    svm_train_time,
    svm_test_time,
    bagging_train_accuracy,
    bagging_test_accuracy,
    bagging_train_time,
    bagging_test_time,
    boosting_train_accuracy,
    boosting_test_accuracy,
    boosting_train_time,
    boosting_test_time
):
    """เปรียบเทียบความแม่นยำ (Train/Test) และเวลาที่ใช้ (Train/Test) ของ SVM ปกติ, Bagging และ Boosting"""

    # จัดเตรียมชื่อโมเดลและค่าผลลัพธ์ของแต่ละโมเดล
    model_names = ["SVM (Baseline)", "Bagging", "Boosting"]
    train_accuracies = [
        svm_train_accuracy * 100.0,
        bagging_train_accuracy * 100.0,
        boosting_train_accuracy * 100.0
    ]
    test_accuracies = [
        svm_test_accuracy * 100.0,
        bagging_test_accuracy * 100.0,
        boosting_test_accuracy * 100.0
    ]
    train_times = [
        svm_train_time,
        bagging_train_time,
        boosting_train_time
    ]
    test_times = [
        svm_test_time,
        bagging_test_time,
        boosting_test_time
    ]

    # แสดงตารางสรุปตัวเลขเปรียบเทียบทั้งความแม่นยำ, เวลาเทรน และเวลาเทส
    print("=" * 85)
    print(f"{'Model':<18} | {'Train Acc (%)':>13} | {'Test Acc (%)':>12} | {'Train Time (s)':>14} | {'Test Time (s)':>14}")
    print("-" * 85)
    for name, train_acc, test_acc, t_train, t_test in zip(
        model_names, train_accuracies, test_accuracies, train_times, test_times
    ):
        print(f"{name:<18} | {train_acc:>12.2f}% | {test_acc:>11.2f}% | {t_train:>13.4f}s | {t_test:>13.6f}s")
    print("=" * 85)

    # สร้างกราฟเปรียบเทียบ 3 ช่อง: ซ้าย = ความแม่นยำ (%), กลาง = เวลาเทรน (s), ขวา = เวลาเทส (s)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # กราฟซ้าย: เปรียบเทียบ Train Accuracy (%) และ Test Accuracy (%)
    x_positions = np.arange(len(model_names))
    bar_width = 0.35

    bars_train = axes[0].bar(
        x_positions - bar_width / 2,
        train_accuracies,
        width=bar_width,
        color="steelblue",
        label="Train Accuracy (%)"
    )
    bars_test = axes[0].bar(
        x_positions + bar_width / 2,
        test_accuracies,
        width=bar_width,
        color="seagreen",
        label="Test Accuracy (%)"
    )

    for bar in bars_train:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    for bar in bars_test:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    axes[0].set_title("Accuracy: SVM vs Bagging vs Boosting", fontsize=12)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 115)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # กราฟกลาง: เปรียบเทียบเวลาที่ใช้ในการเทรน (Training Time)
    time_colors = ["slategray", "darkorange", "crimson"]
    bars_train_time = axes[1].bar(model_names, train_times, width=0.45, color=time_colors)

    max_train_time = max(train_times) if max(train_times) > 0 else 1.0
    for bar in bars_train_time:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_train_time * 0.02),
            f"{height:.4f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    axes[1].set_title("Train Time: SVM vs Bagging vs Boosting", fontsize=12)
    axes[1].set_ylabel("Train Time (seconds)")
    axes[1].set_ylim(0, max_train_time * 1.18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # กราฟขวา: เปรียบเทียบเวลาที่ใช้ในการทำนายชุด Test (Test Time)
    bars_test_time = axes[2].bar(model_names, test_times, width=0.45, color=time_colors)

    max_test_time = max(test_times) if max(test_times) > 0 else 1.0
    for bar in bars_test_time:
        height = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_test_time * 0.02),
            f"{height:.6f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    axes[2].set_title("Test Time: SVM vs Bagging vs Boosting", fontsize=12)
    axes[2].set_ylabel("Test Time (seconds)")
    axes[2].set_ylim(0, max_test_time * 1.18)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
