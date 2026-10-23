import numpy as np
from scipy.stats import spearmanr


def regression_report(y_true, y_pred):
    """All values are in log10(downloads) space."""
    err = y_pred - y_true
    ss_res, ss_tot = (err ** 2).sum(), ((y_true - y_true.mean()) ** 2).sum()
    return {
        "rmse_log10": float(np.sqrt((err ** 2).mean())),
        "mae_log10": float(np.abs(err).mean()),
        "r2": float(1 - ss_res / ss_tot),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
        "within_2x": float((np.abs(err) < np.log10(2)).mean()),
        "within_5x": float((np.abs(err) < np.log10(5)).mean()),
    }
