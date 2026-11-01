import numpy as np
import shap
import matplotlib.pyplot as plt

class ShapAnalysis:

    def __init__(self):
        pass

    def explain_keras(self, model, x, n_background=20, seed=0):
        rng = np.random.default_rng(seed)
        bg = x[rng.choice(len(x), min(n_background, len(x)), replace=False)]
        sv = shap.GradientExplainer(model, bg).shap_values(x)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.shape[-1] == 1:                  # some SHAP versions add an output axis
            sv = sv[..., 0]
        return sv.sum(axis=-1)                 # (N, H, W): signed, summed over colour channels


    def plot_mean_maps(self, heat, x, y, labels=('class 0', 'class 1')):
        sil = (x[..., 0] > 0).astype(float)    # white player pixels are positive after ResNet preprocessing
        lim = np.abs(heat).max()
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        for ax, cls, name in zip(axes, (0, 1), labels):
            m = y == cls
            ax.imshow(1 - sil[m].mean(0), cmap='gray', alpha=0.4)
            ax.imshow(heat[m].mean(0), cmap='bwr', vmin=-lim, vmax=lim, alpha=0.7)
            ax.set_title(f'{name} (n={m.sum()})')
            ax.axis('off')
        plt.show()


    def explain_one(self, model, x, i, nsamples=30, n_background=10, seed=0):
        """SHAP map for the single image x[i]. Returns an (H, W) array."""
        rng = np.random.default_rng(seed)
        bg = x[rng.choice(len(x), min(n_background, len(x)), replace=False)]
        sv = shap.GradientExplainer(model, bg).shap_values(x[i:i + 1], nsamples=nsamples)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.shape[-1] == 1:                  # some SHAP versions add an output axis
            sv = sv[..., 0]
        return sv.sum(axis=-1)[0]              # signed, summed over colour channels


    def plot_one(self, heat, x_i, title=''):
        sil = (x_i[..., 0] > 0).astype(float)  # white player pixels are positive after ResNet preprocessing
        lim = np.abs(heat).max()
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        axes[0].imshow(sil, cmap='gray')
        axes[0].set_title('silhouette')
        axes[1].imshow(1 - sil, cmap='gray', alpha=0.4)
        axes[1].imshow(heat, cmap='bwr', vmin=-lim, vmax=lim, alpha=0.7)
        axes[1].set_title(title)
        for ax in axes:
            ax.axis('off')
        plt.show()