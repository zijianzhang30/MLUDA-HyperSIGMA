"""Run the unmodified Houston baseline and audit per-class metrics.

This wrapper monkey-patches sklearn.metrics.confusion_matrix only to report
the labels/predictions passed by MLUDA_hu.py. Training/configuration is not
changed.
"""

import runpy
import numpy as np
from sklearn import metrics


_orig_confusion_matrix = metrics.confusion_matrix
_orig_kappa = metrics.cohen_kappa_score
_run_idx = 0


def _checked_confusion_matrix(y_true, y_pred, *args, **kwargs):
    global _run_idx
    # Preserve the exact call used by the baseline.
    cm = _orig_confusion_matrix(y_true, y_pred, *args, **kwargs)

    # Force the expected seven post-offset labels for the audit only.
    cm7 = _orig_confusion_matrix(y_true, y_pred, labels=np.arange(7))
    row_totals = cm7.sum(axis=1)
    per_class = np.divide(
        np.diag(cm7), row_totals,
        out=np.full(7, np.nan, dtype=np.float64),
        where=row_totals != 0,
    )
    aa = float(np.nanmean(per_class))
    oa = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    kappa = float(_orig_kappa(y_true, y_pred))
    raw_labels = sorted(set(np.asarray(y_true).astype(int).tolist()))
    pred_labels = sorted(set(np.asarray(y_pred).astype(int).tolist()))
    print(
        "[METRIC_CHECK] run={} raw_true_labels={} pred_labels={} cm_shape={} "
        "OA_eval={:.8f} AA_manual={:.8f} Kappa={:.8f}".format(
            _run_idx, raw_labels, pred_labels, tuple(cm.shape), oa, aa, kappa
        ),
        flush=True,
    )
    print(
        "[METRIC_CHECK] run={} per_class=".format(_run_idx)
        + ",".join("{:.8f}".format(x) for x in per_class),
        flush=True,
    )
    print(
        "[METRIC_CHECK] run={} row_counts=".format(_run_idx)
        + ",".join(str(int(x)) for x in row_totals),
        flush=True,
    )
    _run_idx += 1
    return cm


metrics.confusion_matrix = _checked_confusion_matrix
runpy.run_path("MLUDA_hu.py", run_name="__main__")
