import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import CAT_COLS
from .metrics import regression_report


def run_baselines(xc_tr, xn_tr, y_tr, xc_te, xn_te, y_te):
    res = {"mean_predictor": regression_report(y_te, np.full_like(y_te, y_tr.mean()))}
    X_tr = np.hstack([xc_tr, xn_tr]).astype(np.float32)
    X_te = np.hstack([xc_te, xn_te]).astype(np.float32)
    mask = [True] * len(CAT_COLS) + [False] * xn_tr.shape[1]
    gb = HistGradientBoostingRegressor(categorical_features=mask, max_iter=400,
                                       learning_rate=0.05, random_state=0)
    gb.fit(X_tr, y_tr)
    res["gradient_boosting"] = regression_report(y_te, gb.predict(X_te))
    return res
