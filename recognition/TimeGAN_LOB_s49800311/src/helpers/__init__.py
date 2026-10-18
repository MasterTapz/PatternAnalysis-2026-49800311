"""Support code for train.py and predict.py.

metrics      validity rates, KL divergences and VWAP on decoded books (PyTorch only)
checkpoints  model registry, run names, saving and loading checkpoints
evaluation   validation scoring during training and the test-hour evaluation of best.pt
audit        the predict.py analysis: SSIM, volatility clustering, price paths, VWAP, diversity
plotting     training curves, heatmap autopsies, volatility plots and the README assets
"""
