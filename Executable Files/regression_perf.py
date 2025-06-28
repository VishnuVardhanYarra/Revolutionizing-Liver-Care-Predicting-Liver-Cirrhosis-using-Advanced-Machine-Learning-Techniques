import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def regression_performance(y_true, y_pred, plot_dir=None, tag="model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_true, y_pred)

    scores = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        # ✅ Fixed heatmap generation
        score_df = pd.DataFrame([scores])  # Proper 1x4 DataFrame
        plt.figure(figsize=(6, 2))
        sns.heatmap(score_df, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"{tag} - Regression Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{tag}_metrics_heatmap.png"))
        plt.close()

        # Parity plot
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{tag} - Parity Plot")
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{tag}_parity.png"))
        plt.close()

    return scores
