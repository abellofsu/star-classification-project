# star-classication-project
Project for FSU CIS4930 Data Mining

# correlations

![ASAS-SN Feature Correlations](pictures/correlations.png)


V and g magnitudes: 0.98

V and g amplitudes: 0.93

BP-RP and J-K color indices: 0.91

g-band log-period and J-K: 0.82

# pca


| Feature | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_mag | -0.189 | 0.319 | 0.514 | 0.094 | -0.211 | 0.140 | 0.066 | 0.070 |
| g_mag | -0.103 | 0.331 | 0.541 | 0.132 | -0.278 | 0.117 | 0.025 | 0.030 |
| v_amplitude | 0.119 | 0.548 | -0.133 | -0.313 | 0.153 | -0.192 | 0.070 | 0.041 |
| g_amplitude | 0.119 | 0.549 | -0.090 | -0.341 | 0.212 | -0.142 | 0.031 | -0.056 |
| v_log_period | 0.409 | -0.028 | 0.025 | -0.213 | 0.053 | 0.761 | 0.276 | -0.354 |
| g_log_period | 0.454 | -0.052 | -0.050 | -0.077 | -0.235 | 0.074 | 0.181 | 0.823 |
| v_lksl | 0.171 | -0.180 | 0.483 | -0.201 | 0.585 | 0.064 | -0.530 | 0.184 |
| g_lksl | 0.265 | -0.250 | 0.399 | -0.000 | 0.197 | -0.490 | 0.631 | -0.178 |
| bp_rp | 0.455 | 0.077 | -0.007 | 0.109 | -0.276 | -0.195 | -0.321 | -0.298 |
| j_k | 0.455 | 0.011 | 0.070 | 0.128 | -0.310 | -0.177 | -0.303 | -0.165 |
| w1_w2 | 0.197 | 0.287 | -0.127 | 0.800 | 0.446 | 0.109 | 0.079 | 0.078 |


the cnn will use time series data, but for the mlp we decide inputs with this ^^^


# dataset splitting
divide the matched stars into training, validation, and testing datasets using an approximately 70/15/15 stratified split

998 stars with competing classifications: set aside until we figure out what to do


# cnn (light curves, cross-band)

`python cnn.py` runs everything in one go: reads and folds the `.dat` light curves, draws the light-curve figures, trains MLP vs CNN vs CNN+FGSM on each band with 5 seeds, tests every model on both bands (clean, with Gaussian noise, and under FGSM attack), and saves tables and figures. Progress and an ETA are printed after every training run.

Estimated time: about 6 minutes on one GPU (measured: 5 min 52 s on an FSU RCC A4500), longer on a laptop CPU. Libraries: `torch numpy pandas scikit-learn matplotlib`.

**Input.** Each star's `.dat` light curve, phase-folded on its catalog period and median-binned into 128 bins (magnitude minus the star's median), plus log10(period) and a has-period flag. Irregular `L` stars have no period, so their curve is binned over time instead.

**Light-curve cleaning (the "some .dat files have no light curves" issue).** Upper limits like `>17.4`, rows without a magnitude, and points with `mag_err = 99.99` are removed. A star needs at least 20 good points in BOTH bands, so every model and band pair is scored on the same stars. 90 V-band files are header only (no observations); those stars are dropped. Every dropped star and its reason is listed in `results/cnn_dropped_stars.csv`.

**Classes.** The 12 types that appear in val and test (DCEP, DSCT, EA, EB, EW, HADS, L, M, ROT, RRAB, RRC, SR). Types with fewer than 7 stars sit in train only, so they cannot be scored and are left out.

**Models** (same input for all three):
- MLP: 130 inputs -> 128 -> 64 -> 32 -> 12, ReLU, dropout 0.2
- CNN (Aiden's design): Conv1d(32 filters, k=9) -> MaxPool(2) -> Conv1d(64, k=9) -> global average pool -> dense 64 (+ period inputs) -> 12-way softmax. Circular padding because a folded curve wraps around.
- CNN+FGSM: same CNN, loss = 0.5 x clean + 0.5 x adversarial, with `x_adv = x + eps * sign(grad_x L)` and eps = 0.05 mag

**Training.** Adam (lr 1e-3, weight decay 1e-4), class-balanced cross-entropy, batch 128, early stopping on validation macro-F1 in the training band only (the other band is never seen until test).

**Outputs.**
- `results/cnn_summary.csv`: accuracy and macro-F1 (mean ± std over 5 seeds) for every model and train -> test band
- `results/cnn_per_class.csv`, `results/cnn_all_runs.csv` (noise and FGSM levels), `results/cnn_training.csv`
- `pictures/cnn_lightcurve_examples.png`, `pictures/cnn_class_median_curves.png`: what the light curves look like
- `pictures/cnn_cross_band_f1.png`, `pictures/cnn_noise_robustness.png`, `pictures/cnn_fgsm_robustness.png`, `pictures/cnn_confusion_cross_band.png`

**Results** (test set, 595 stars, mean ± std over 5 seeds, macro-F1 / accuracy):

| model | V -> V | V -> G | G -> G | G -> V |
|---|---|---|---|---|
| MLP | 0.908 / 0.929 | 0.834 / 0.914 | 0.860 / 0.916 | 0.863 / 0.906 |
| CNN | 0.906 / 0.941 | 0.824 / 0.928 | 0.848 / 0.910 | 0.809 / 0.877 |
| CNN+FGSM | 0.893 / 0.906 | 0.853 / 0.901 | 0.857 / 0.885 | 0.814 / 0.871 |

- On clean curves all three models are close (within about one std). The CNN has the best accuracy in the V band (0.941). CNN+FGSM has the best cross-band macro-F1 from V to G (0.853).
- Adversarial training is where the difference shows. Under an FGSM attack of 0.05 mag, macro-F1 (V -> V) is 0.70 for CNN+FGSM, 0.48 for the CNN, and 0.32 for the MLP (`pictures/cnn_fgsm_robustness.png`).
- Random Gaussian noise hurts the plain CNN most. At 0.2 mag, V -> V F1 is 0.68 for the MLP, 0.60 for CNN+FGSM, and 0.49 for the CNN (`pictures/cnn_noise_robustness.png`).
- Main cross-band error: irregular `L` stars trained on V are called `SR` in G. In the V catalog 0 of 31 training L stars have a period; in the G catalog 24 of 31 do. So in G they get folded like semi-regulars. That is a catalog difference between the bands, not just noise (`pictures/cnn_confusion_cross_band.png`).
