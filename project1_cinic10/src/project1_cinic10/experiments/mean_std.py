import numpy as np

results = {
    "1-shot": {
        "test/acc": [0.4188, 0.444, 0.4764],
        "test/loss": [1.32561, 1.26464, 1.23525],
    },
    "5-shot": {
        "test/acc": [0.6204, 0.6324, 0.6496],
        "test/loss": [0.91444, 0.9558, 0.8568],
    },
    "10-shot": {
        "test/acc": [0.6636, 0.6852, 0.6752],
        "test/loss": [0.86812, 0.80951, 0.82447],
    },
}

for shot, metrics in results.items():
    acc = np.array(metrics["test/acc"], dtype=float)
    loss = np.array(metrics["test/loss"], dtype=float)

    acc_mean = np.mean(acc)
    acc_std = np.std(acc)

    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    print(f"{shot}")
    print(f"  test/acc  = {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  test/loss = {loss_mean:.4f} ± {loss_std:.4f}")
    print()