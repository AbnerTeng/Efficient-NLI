"""
Evaluation functions for BERTNLI.
"""
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize
from .train import pred


def eval_model(model, data_loader, y, device):
    """
    evaluate model
    """
    predict_sigmoid = pred(
        model, data_loader, device, proba=True, to_numpy=True
    )[:, 0].reshape((-1, 3))
    loss_s = [
        log_loss(
            y[i::3], predict_sigmoid[:, i]
        )
        for i in range(3)
    ]
    predict_abn = normalize(
        predict_sigmoid, norm='l1'
    )
    abn_loss = log_loss(
        y.reshape((-1, 3)), predict_abn
    )
    return {
        'A_loss': loss_s[0],
        'B_loss': loss_s[1],
        'N_loss': loss_s[2],
        'abn_loss': abn_loss
    }
