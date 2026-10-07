"""Support code for train.py and predict.py.

metrics      validity rates and KL divergences on decoded books (PyTorch only)
checkpoints  model registry, run names, saving and loading checkpoints
evaluation   validation scoring during training and the test-hour evaluation of best.pt
audit        the predict.py analysis: SSIM, volatility clustering, price paths, diversity
plotting     training curves, heatmap autopsies and volatility plots
"""
