import numpy as np
import torch

# Test Label Relaxation loss
from label_smoothing import LabelRelaxationLoss

lr_alpha = 0.25
lr_loss = LabelRelaxationLoss(lr_alpha, logits_provided=False, one_hot_encode_trgts=False)

y_true = np.array([1., 0., 0.], dtype=np.float32)
y_pred = np.array([0.675, 0.175, 0.15], dtype=np.float32)
exp_result = 0.01342932

np.testing.assert_almost_equal(lr_loss.forward(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy(),
                               exp_result)

y_true = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], dtype=np.float32)
y_pred = np.array([[0.675, 0.175, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15]], dtype=np.float32)
exp_result = (0.01342932 + 0.006164 * 3) / 4

np.testing.assert_almost_equal(lr_loss.forward(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy(),
                               exp_result, decimal=4)

# Test one-hot encoding variant
lr_loss = LabelRelaxationLoss(lr_alpha, logits_provided=False, one_hot_encode_trgts=True, num_classes=3)

y_true = np.array([0], dtype=np.int)
y_pred = np.array([0.675, 0.175, 0.15], dtype=np.float32)
exp_result = 0.01342932

np.testing.assert_almost_equal(lr_loss.forward(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy(),
                               exp_result)

y_true = np.array([0, 1, 1, 1], dtype=np.int)
y_pred = np.array([[0.675, 0.175, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15]], dtype=np.float32)
exp_result = (0.01342932 + 0.006164 * 3) / 4

np.testing.assert_almost_equal(lr_loss.forward(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy(),
                               exp_result, decimal=4)
