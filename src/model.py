"""Model utilities for student attrition prediction.

Split strategy: id_student determines train/val/test membership.
Same student cannot appear in both train and test across different course enrollments.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

KEY_COLS = ['code_module', 'code_presentation', 'id_student']


def make_student_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split by unique id_student so same student never spans train/test splits.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    X must have id_student as column or index level.
    """
    X = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()

    unique_ids = X['id_student'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test:n_test + n_val])
    train_ids = set(shuffled[n_test + n_val:])

    train_mask = X['id_student'].isin(train_ids)
    val_mask = X['id_student'].isin(val_ids)
    test_mask = X['id_student'].isin(test_ids)

    feat_cols = [c for c in X.columns if c not in KEY_COLS]

    X_train = X.loc[train_mask, feat_cols].reset_index(drop=True)
    X_val   = X.loc[val_mask,   feat_cols].reset_index(drop=True)
    X_test  = X.loc[test_mask,  feat_cols].reset_index(drop=True)

    y_reset = y.reset_index(drop=True) if not isinstance(y, pd.Series) else y
    if len(y_reset) != len(X):
        y_reset = y.values

    y_train = pd.Series(y_reset).iloc[train_mask.values].reset_index(drop=True)
    y_val   = pd.Series(y_reset).iloc[val_mask.values].reset_index(drop=True)
    y_test  = pd.Series(y_reset).iloc[test_mask.values].reset_index(drop=True)

    print(f'Train: {len(X_train):,} rows ({y_train.mean()*100:.1f}% positive)')
    print(f'Val:   {len(X_val):,}   rows ({y_val.mean()*100:.1f}% positive)')
    print(f'Test:  {len(X_test):,}  rows ({y_test.mean()*100:.1f}% positive)')

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """Return evaluation metrics dict. Uses predict_proba if available."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'f1_minority':  round(f1_score(y, pred, pos_label=1), 4),
        'roc_auc':      round(roc_auc_score(y, proba), 4),
        'precision':    round(precision_score(y, pred, pos_label=1, zero_division=0), 4),
        'recall':       round(recall_score(y, pred, pos_label=1, zero_division=0), 4),
        'fpr':          round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def get_tree_explainer(model) -> shap.TreeExplainer:
    """Return TreeExplainer, unwrapping CalibratedClassifierCV and imblearn Pipeline."""
    base = model
    # Unwrap CalibratedClassifierCV
    if hasattr(base, 'calibrated_classifiers_'):
        base = base.calibrated_classifiers_[0].estimator
    elif hasattr(base, 'estimator'):
        base = base.estimator
    # Unwrap imblearn/sklearn Pipeline to get the final estimator step
    if hasattr(base, 'steps'):
        base = base.steps[-1][1]
    return shap.TreeExplainer(base)


def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for positive class. Returns array shape (n_rows, n_features)."""
    explainer = get_tree_explainer(model)
    sv = explainer.shap_values(X)
    # XGBoost TreeExplainer returns array directly (not list) for binary classification
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def top_shap_features(shap_values: np.ndarray, feature_names: list, n: int = 3) -> pd.DataFrame:
    """For each row, return top n features by |SHAP value| with direction.

    Returns DataFrame with columns: top_shap_feature_{1..n}, top_shap_value_{1..n}
    """
    rows = []
    for row_sv in shap_values:
        ranked = sorted(zip(feature_names, row_sv), key=lambda x: abs(x[1]), reverse=True)[:n]
        flat = {}
        for i, (feat, val) in enumerate(ranked, 1):
            flat[f'top_shap_feature_{i}'] = feat
            flat[f'top_shap_value_{i}'] = round(float(val), 4)
        rows.append(flat)
    return pd.DataFrame(rows)


def build_predictions_csv(
    model,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    info: pd.DataFrame,
    save_path: str | Path = 'data/output/predictions.csv',
) -> pd.DataFrame:
    """Generate predictions.csv: risk_score, predicted_label, true_label, top-3 SHAP per student."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in X_full.columns if c not in KEY_COLS]
    X = X_full[feat_cols] if isinstance(X_full, pd.DataFrame) else X_full

    proba = model.predict_proba(X)[:, 1]
    pred_label = (proba >= 0.5).astype(int)
    risk_score = (proba * 100).round(1)

    print('Computing SHAP values (this may take ~30s)...')
    shap_vals = compute_shap(model, X)
    shap_df = top_shap_features(shap_vals, feat_cols)

    out = info[KEY_COLS].reset_index(drop=True).copy()
    out['risk_score'] = risk_score
    out['predicted_label'] = pred_label
    out['true_label'] = y_full.reset_index(drop=True).values
    out = pd.concat([out, shap_df], axis=1)

    out.to_csv(save_path, index=False)
    print(f'Saved {len(out):,} predictions to {save_path}')
    return out


def load_model(path: str | Path = 'models/retention_model.pkl'):
    return joblib.load(path)
