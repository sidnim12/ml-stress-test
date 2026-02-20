from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score
)

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]
MetricFn = Callable[..., float]


def _to_1d(a: ArrayLike) -> np.ndarray:
    """convert to 1d numpy array"""
    
    arr = np.asarray(a)
    
    if arr.ndim == 0:
        return arr.reshape(1)
    
    if arr.ndim == 1:
        return arr
    
    # If shape (n,1) -> flatten; if (n,k) keep as-is (probabilities)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    
    return arr
        
        
        
def _is_probabilities(y_pred: np.ndarray) -> bool:
    """Heuristic: probabilities are 2D (n, k) or 1D float in [0,1]."""
    
    if y_pred.ndim == 2:
        return True
    
    if y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating):
        # allow logits sometimes, but we can’t reliably detect; keep it simple
        return np.nanmin(y_pred) >= 0.0 and np.nanmax(y_pred) <= 1.0
    



def _safe_metric(
    fn: Callable[..., float],
    *,
    strict: bool = False,
    default: float = float("nan"),
    name: str = "metric",  
) -> Callable[..., float]:
    """Wrap a metric to avoid crashing web routes; optionally raise in strict mode."""
    def wrapped(*args: Any, **kwargs: Any ) -> float:
            try: 
                val = fn(*args, **kwargs)
                return float(val)
            
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to compute {name}: {e}") from e
                return float(default)
    return wrapped


@dataclass(frozen=True)
class ClassificationConfig:
    average: str = "weighted"         # weighted/macro/micro
    pos_label: int = 1                # used for binary precision/recall if needed
    auc_multi_class: str = "ovr"      # 'ovr' or 'ovo' when multiclass
    auc_average: str = "weighted"     
    strict: bool = False              
    zero_division: int = 0            


@dataclass(frozen=True)
class RegressionConfig:
    strict: bool = False
    default_on_fail: float = float("nan")
    mape_epsilon: float = 1e-8        # avoid divide-by-zero
    



    

# CLASSIFICATION METRICS :-

def classification_metrics(cfg: Optional[ClassificationConfig] = None) -> Dict[str, MetricFn]:
    cfg = cfg or ClassificationConfig()
    
    def _accuracy(y_true: ArrayLike, y_pred:ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return accuracy_score(yt, yp)
    
    
    def _f1(y_true: ArrayLike, y_pred_labels: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred_labels)
        return f1_score(yt, yp, average=cfg.average)
    
    
    def _balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return balanced_accuracy_score(yt, yp)
    
    
    def _precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return precision_score(yt, yp, average=cfg.average, zero_division=cfg.zero_division)

    
    def _recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return recall_score(yt, yp, average=cfg.average, zero_division=cfg.zero_division)


    
    def _roc_auc(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
        """
        Binary: y_pred_proba can be (n,) prob for class 1 or (n,2) probs.
        Multiclass: y_pred_proba should be (n,k).
        """
        yt = _to_1d(y_true)
        yp = np.asarray(y_pred_proba)
        
        
        # If user passed labels by mistake, fail safely
        if not _is_probabilities(np.asarray(yp)):
            raise ValueError("roc_auc expects probabilities, got labels/unknown format")
        
        
        #Binary case
        if yp.ndim ==1:
            return roc_auc_score(yt, yp)  
        
        # (n,k) probabilities
        k = yp.shape[1]
        if k == 2:
            return roc_auc_score(yt, yp[:, 1])
        
        # Multiclass
        return roc_auc_score(
            yt,
            yp,
            multi_class=cfg.auc_multi_class,
            average=cfg.auc_average,
        )
    


    def _pr_auc(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
        """
        Average precision (PR-AUC). Binary-only unless you do one-vs-rest yourself.
        """
        yt = _to_1d(y_true)
        yp = np.asarray(y_pred_proba)

        if yp.ndim == 2 and yp.shape[1] == 2:
            yp = yp[:, 1]  # positive class prob

        if yp.ndim != 1:
            raise ValueError("average_precision expects binary (n,) probabilities")

        return average_precision_score(yt, yp)
        
        
        
    def _logloss(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = np.asarray(y_pred_proba)
        
        if yp.ndim == 1:
            yp = np.column_stack([1-yp, yp])
            return log_loss(yt, yp)
    
    return {
        # label-based
        "accuracy": _safe_metric(_accuracy, strict=cfg.strict, name="accuracy"),
        "balanced_accuracy": _safe_metric(_balanced_accuracy, strict=cfg.strict, name="balanced_accuracy"),
        "f1": _safe_metric(_f1, strict=cfg.strict, name="f1"),
        "precision": _safe_metric(_precision, strict=cfg.strict, name="precision"),
        "recall": _safe_metric(_recall, strict=cfg.strict, name="recall"),
        
        
        # probability-based
        "roc_auc": _safe_metric(_roc_auc, strict=cfg.strict, name="roc_auc"),
        "pr_auc": _safe_metric(_pr_auc, strict=cfg.strict, name="pr_auc"),
        "log_loss": _safe_metric(_logloss, strict=cfg.strict, name="log_loss"),
    }

  
  
    
    
# REGRESSION METRICS

def regression_metrics(cfg: Optional[RegressionConfig] = None) -> Dict[str, MetricFn]:
    cfg = cfg or RegressionConfig()

    def _rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return root_mean_squared_error(yt, yp, squared=False)


    def _mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return mean_absolute_error(yt, yp)


    def _median_ae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return median_absolute_error(yt, yp)


    def _r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return r2_score(yt, yp)


    def _explained_var(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true)
        yp = _to_1d(y_pred)
        return explained_variance_score(yt, yp)


    def _mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true).astype(float)
        yp = _to_1d(y_pred).astype(float)
        denom = np.maximum(np.abs(yt), cfg.mape_epsilon)
        return float(np.mean(np.abs((yt - yp) / denom))) * 100.0
    

    def _smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = _to_1d(y_true).astype(float)
        yp = _to_1d(y_pred).astype(float)
        denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, cfg.mape_epsilon)
        return float(np.mean(np.abs(yt - yp) / denom)) * 100.0
    
    
    def wrap(name: str, fn: Callable[..., float]) -> Callable[..., float]:
        return _safe_metric(fn, strict=cfg.strict, default=cfg.default_on_fail, name=name)
    
    
    return {
        "rmse": wrap("rmse", _rmse),
        "mae": wrap("mae", _mae),
        "median_ae": wrap("median_ae", _median_ae),
        "r2": wrap("r2", _r2),
        "explained_variance": wrap("explained_variance", _explained_var),
        "mape_percent": wrap("mape_percent", _mape),
        "smape_percent": wrap("smape_percent", _smape),
    }




def compute_metrics(
    metrics: Dict[str, MetricFn],
    *,
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_proba: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    
    """
    Offers Webdev-friendly convenience:
    - For label metrics pass y_pred
    - For proba metrics pass y_proba
    It will compute whatever it can and return a dict of floats.
    """
    
    out: Dict[str, float] = {}
    
    for name, fn in metrics.items():
        
        if name in {"roc_auc", "pr_auc", "log_loss"}:
            if y_proba is None:
                out[name] = float("nan")
            else:
                out[name] = fn(y_true, y_proba)
        
        else:
            out[name] = fn(y_true, y_pred)
    return out
    
    