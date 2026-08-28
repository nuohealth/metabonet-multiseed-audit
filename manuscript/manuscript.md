# Multi-seed robustness auditing of recurrent architectures for blood glucose forecasting

**Running title:** Multi-seed audit of glucose forecasting

Zhongwei Sun¹*, Yongping Ma², Xinran Wu³, Ping Wang⁴

1 Institute of Exercise Science, Guangzhou College of Technology and Business, Guangzhou, China
2 Department of Physical Education and Science, Xinjiang University, Urumqi, China
3 Department of Physical Education, Sun Yat-sen University, Guangzhou, China
4 Institute of Sports Science, Guangxi University of Science and Technology, Liuzhou, China

* Corresponding author: Zhongwei Sun, Email: sunzhongwei@gzgs.edu.cn

---

## Abstract

Blood glucose (BG) forecasting from continuous glucose monitoring (CGM) data is a key component of decision support in type 1 diabetes (T1D) management. Whether increasing the architectural complexity of deep recurrent models yields meaningful accuracy gains on large-scale, real-world CGM data remains unclear, in part because most comparative studies train each model only once, conflating true architectural differences with the noise of stochastic initialisation. Here we evaluate this question with a multi-seed robustness-auditing protocol: four recurrent architectures (a plain LSTM, a bidirectional LSTM, an attention-augmented LSTM, and a bidirectional attention LSTM) were trained under an identical pipeline with twenty random seeds each (n = 20) on the public MetaboNet resource — the largest consolidated T1D CGM data set (3,135 subjects; 1,092 met our quality filters and entered the analysis) — and on an independent T2D cross-cohort replication set. All models performed almost identically (mean RMSE 16.23–16.30 mg/dL; R2 0.9236–0.9242), with the largest between-architecture difference (0.065 mg/dL) being 77-fold smaller than a clinically meaningful threshold, and with within-architecture between-seed variability of the same order as the between-architecture differences. A plain, parameter-heavy LSTM matched the most complex architecture. Conclusions persisted in the independent T2D cohort. These results show that, for short-horizon, CGM-only glucose forecasting, added architectural complexity yields no clinically meaningful accuracy gain once between-seed variability is properly accounted for, and that single-seed comparisons are insufficient for model selection. All models and code are released to support reproducible benchmarking in medical time-series AI.

**Keywords:** Continuous glucose monitoring; Blood glucose prediction; Recurrent neural network; Attention mechanism; Model robustness; Type 1 diabetes

---

## 1 Introduction

### 1.1 Background

Diabetes mellitus is a chronic metabolic disorder characterised by hyperglycaemia, affecting approximately 537 million adults worldwide as of 2021, with projections rising to 643 million by 2030 and 783 million by 2045 [1]. Poor glycaemic control predisposes patients to microvascular complications — retinopathy, nephropathy, and neuropathy — as well as macrovascular events such as cardiovascular disease and stroke [2, 3].

Continuous glucose monitoring (CGM) technology has revolutionised diabetes management by providing real-time subcutaneous interstitial glucose measurements at 1–5 minute intervals [2]. Unlike finger-stick measurements producing isolated snapshots, CGM generates up to 288 data points per day, capturing postprandial excursions, nocturnal trends, and asymptomatic hypoglycaemic episodes [2, 4]. Modern CGM systems demonstrate MARD values below 10% and Clarke Error Grid A+B zone rates exceeding 98% [2]. Machine-learning approaches have further shown the ability to predict time-in-range metrics and HbA1c from CGM-derived features, underscoring the clinical value of CGM data [6].

Reliable short-term blood glucose (BG) prediction offers concrete clinical benefits: early warnings for hypoglycaemia (15–30 min ahead) enable preemptive carbohydrate intake and reduce severe events [3]; predictive models inform insulin-pump adjustment within closed-loop artificial pancreas systems [1, 3]; forecasts empower lifestyle decisions [2, 4]; and patient-specific patterns support individualised treatment. Exercise-induced glycaemic fluctuations, including rapid drops and late-onset post-exercise hypoglycaemia, pose additional prediction challenges that benefit from CGM-informed models [2].

### 1.2 Related Work and the Open Question

Conventional BG-prediction approaches include autoregressive integrated moving average (ARIMA) and physiological compartment models [5]; while interpretable, these linear methods struggle with the nonlinear dynamics of glucose regulation [14]. The advent of deep learning introduced recurrent network (RNN)-based solutions. Li et al. predicted 30-minute BG with a convolutional recurrent network (RMSE ≈18.7 mg/dL) [7]; Zhu et al. applied an evidential deep learning approach for personalised T1D prediction [8]; Mirshekarian et al. showed recurrent models outperform support vector regression and random forests [9]. Transformer-based architectures have been applied at multiple horizons with competitive results [13], though their application to CGM remains limited by computational demands and scarce large-scale datasets [13, 14].

The attention mechanism — originally proposed for neural machine translation [11] and since adapted to time series [12] — has been widely reported to improve medical time-series modelling, including BG prediction [10, 13]. Attention enables models to focus on critical historical time points such as pre-prandial patterns or rapid glycaemic shifts [10]. Similarly, bidirectional recurrent processing captures both antecedent and consequent context within an input window.

**However, a critical methodological gap persists.** Most comparative studies in CGM-based glucose forecasting evaluate a single random seed or use non-uniform hyperparameters across architectures, making it impossible to determine whether reported advantages (e.g., an "attention gain") reflect the architecture itself or incidental training conditions. Robustness auditing — averaging across multiple seeds — is essential to distinguish genuine architectural effects from sampling noise [17, 20]. On small cohorts, conclusions may further be driven by heterogeneity rather than model capacity. Whether architectural complexity yields *clinically meaningful, statistically robust* gains on **large-scale, real-world, univariate CGM** data has not been systematically established.***


Relation to concurrent benchmark initiatives. Two closely related efforts frame the contribution of the present study and explain its distinct focus. First, the MetaboNet dataset itself was recently used to build MetaboNet-Bench, a multimodal benchmarking framework that standardises forecasting tasks across CGM, insulin, carbohydrate, and activity signals and provides fixed data splits for fair model comparison [26]. MetaboNet-Bench addresses the question "how should models be compared fairly across modalities". The present study asks a complementary question that MetaboNet-Bench does not directly answer: "given a fixed, univariate CGM-only setting and an identical training pipeline, are between-architecture differences reliable or do they vanish once between-seed variability is accounted for?". Whereas MetaboNet-Bench evaluates models on a single training run per configuration, we vary only the random seed across twenty repeats and report between-seed dispersion, a robustness-auditing dimension that is essential for model selection but has not been systematically quantified for large-scale CGM forecasting. We therefore position this work as a methodological complement to, rather than a duplicate of, existing benchmark initiatives.

### 1.3 Contributions of This Work

This study addresses that gap through a controlled, multi-seed robustness-benchmarking protocol on the largest publicly available consolidated CGM dataset. Our contributions are:

1. **A large-scale, real-world benchmark.** We compare four recurrent architectures (LSTM, BiLSTM, LSTM-Attention, BiLSTM-Attention) on 1,092 persons with T1D from MetaboNet [15] — ≈91-times larger than the single-site OhioT1DM cohort (n = 12) used in prior CGM forecasting studies [7, 9, 13].
2. **A controlled multi-seed robustness-benchmarking protocol.** We fix the training pipeline, windowing, and all hyperparameters; vary only the architecture; and repeat each configuration across twenty independent seeds (n = 20), reporting mean ± SD together with between-seed dispersion. This isolates the architectural effect from optimisation and stochastic-initialisation confounds and provides a reproducible protocol for model benchmarking in medical time-series analysis [17, 20].
3. **Statistical robustness auditing.** We apply paired t-test, Wilcoxon signed-rank, effect size (Cohen's d), Holm–Bonferroni correction for multiple comparisons, and bootstrap confidence intervals, interpreting differences primarily against a clinically meaningful RMSE threshold rather than relying on raw metric deltas alone.
4. **Clinically grounded evaluation.** Beyond regression metrics, we report Clarke error-grid analysis, per-horizon accuracy, stratified RMSE, MARD, and hypoglycaemia detection, distinguishing clinically acceptable short-horizon performance from long-horizon and hypoglycaemia limitations.

---

## 2 Methods

### 2.1 Data Source and Cohort

This study uses the **MetaboNet dataset** [15], the largest publicly available consolidated CGM dataset for T1D management. MetaboNet aggregates multiple clinical studies into a unified, pre-processed repository of CGM recordings at 5-minute median sampling intervals, accessible at https://metabo-net.org; the full cohort and its composition are described in the dataset paper [15]. From the full cohort, we applied a quality-filtering pipeline: inclusion of participants with a T1D diagnosis; a minimum of 7 days of valid CGM recording; at least 18 consecutive hours of valid CGM data after sliding-window generation; and manual review of trace integrity. After filtering, **1,092 subjects** met all criteria and were included in the final analysis. The overall CGM mean was **150.8 mg/dL (SD = 59.8 mg/dL)**, consistent with an expected T1D glycaemic profile.

Data were partitioned at the subject level (60% training / 20% validation / 20% testing) to ensure patient-independent evaluation:
- **Training set:** 655 subjects / 655,000 windows
- **Validation set:** 218 subjects / 218,000 windows
- **Test set:** 219 subjects / 219,000 windows

No data from the same participant appears in more than one partition, simulating deployment to unseen individuals.

> **Note on cohort size.** Prior single-site CGM forecasting cohorts (e.g., OhioT1DM, 12 subjects [23]) are far smaller. Our 1,092-subject cohort yields 1,092,000 sliding windows and 1,314,000 flattened prediction–actual pairs in clinical evaluation, providing substantially greater statistical power and heterogeneity.

**T2D cross-cohort validation dataset.** To assess cross-population generalisability of the robustness-benchmarking conclusions, we additionally validated all four architectures on an independent type 2 diabetes (T2D) CGM cohort. The **Shanghai T2DM** dataset [25] provides real-world, ambulatory CGM recordings from adults with type 2 diabetes followed in a clinic setting; raw sensor traces were obtained from the public figshare repository [25] and processed through an identical preprocessing pipeline (Z-score normalisation, temporal hour/minute encoding, sliding-window generation with $L{=}12$ / $H{=}6$ at 5-min sampling, stride 1) as the primary T1D cohort. From raw recordings with a minimum of 7 days of valid CGM data, **100 unique T2D subjects** with sufficient valid recordings were retained and partitioned into training/validation/testing subjects following the same strict 60/20/20 subject-level split (60 / 20 / 20 subjects, trained/validation/test, respectively) used for T1D. Because the underlying figshare release provides per-subject recordings that may span multiple files (e.g., successive clinic visits of the same patient), all CGM segments of a given patient were first aggregated, and each unique patient was assigned to exactly one partition; this strict subject-disjoint split was verified programmatically, with zero overlap between any pair of partitions (train ∩ val ∩ test = ∅). All sliding windows for a partition therefore derive exclusively from that partition's patients, ensuring fully patient-independent evaluation with no data leakage between training and test sets. This T2D cohort is markedly smaller and, as expected for type 2 diabetes, exhibits different glycaemic dynamics and medication burden than the T1D cohort; it is used exclusively as an external, cross-cohort replication set in which the four architectures are **retrained from scratch** on the T2D data to test whether the primary conclusions — that architectural complexity yields no clinically meaningful gain, and that between-seed variation dominates between-architecture differences — persist across a different disease population. To keep clinical-evaluation scope identical between cohorts, all 20-seed robustness runs, the cross-cohort comparisons, and the training-cost analysis reported in Figures 1–4 were performed on both cohorts under identical protocol.

> **Sliding-window cap.** To bound memory and equalise per-subject representation, each subject's window stream was capped at **1,000 windows per subject** (i.e., the first 1,000 valid sliding windows); this cap, applied identically to both cohorts, explains the exact 1,000-window-per-subject counts reported above, independent of trace length. All windows generated from a subject were contiguous in the original trace, so the cap does not alter the temporal structure of the retained windows.

### 2.2 Problem Formulation

BG prediction is formulated as a multi-step time-series regression. Given a historical observation sequence

$$\mathbf{X} = [x_{t-L+1}, x_{t-L+2}, \dots, x_t]$$

with input window $L = 12$ steps (60 minutes), the model predicts future values

$$\mathbf{Y} = [\hat{x}_{t+1}, \hat{x}_{t+2}, \dots, \hat{x}_{t+H}]$$

with prediction horizon $H = 6$ steps (30 minutes). Each input feature vector $x_t \in \mathbb{R}^3$ comprises the CGM glucose value $g_t$ (mg/dL), the hour of day, and the minute of day, both normalised to a [0,1] range. Overlapping windows (stride = 1) generate input–output pairs $(X_{t-L+1:t}, Y_{t+1:t+H})$.

### 2.3 Model Architectures

We compare four recurrent architectures of increasing nominal complexity while holding the training protocol fixed:

**1. Plain LSTM.** A two-layer LSTM encoder (128 hidden units per layer) followed by a fully connected regression head. The final hidden state is projected to the H-step output.

**2. Bidirectional LSTM (BiLSTM).** Processes the sequence in both forward and backward directions:

$$\overrightarrow{h_t} = \text{LSTM}_{\text{forward}}(x_t, \overrightarrow{h_{t-1}}), \quad \overleftarrow{h_t} = \text{LSTM}_{\text{backward}}(x_t, \overleftarrow{h_{t+1}}), \quad h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

with two stacked layers and 64 hidden units per direction (concatenated to a 128-dimensional output per time step).

**3. LSTM-Attention.** A two-layer LSTM encoder (128 hidden units per layer) augmented with an additive (Bahdanau) attention module [11] that computes a context vector as a weighted sum of hidden states:

$$e_t = \tanh(W_a h_t + b_a), \quad \alpha_t = \frac{\exp(e_t^{\mathsf{T}} w)}{\sum_j \exp(e_j^{\mathsf{T}} w)}, \quad \mathbf{c} = \sum_t \alpha_t h_t$$

where $W_a$, $b_a$, and context vector $w$ are learnable, and $\alpha_t$ is the softmax-normalised attention weight.

**4. BiLSTM-Attention.** Combines the bidirectional encoder (64 hidden units per direction) with the additive attention module described above.

**Shared output head.** For all architectures, the (possibly attention-aggregated) representation is projected to the multi-step output through a two-layer fully connected network with dropout ($p=0.3$) and ReLU activations:

$$\hat{y} = W_3 \cdot \text{ReLU}\big(W_2 \cdot \text{Dropout}(\text{ReLU}(W_1 v + b_1)) + b_2\big) + b_3$$

where $v$ is the final encoder (or context) vector and $\hat{y} \in \mathbb{R}^H$ contains simultaneous predictions for all future steps.

Trainable parameter counts: LSTM 208,838; BiLSTM 143,302; LSTM-Attention 212,998; BiLSTM-Attention 147,462. Notably, the plain LSTM has *more* parameters than the BiLSTM variants — complexity in the sense of architectural mechanism (attention, bidirectionality) rather than raw parameter count.

### 2.4 Training and Evaluation Protocol

**Loss function.** All models minimise the mean squared error (MSE) across all prediction horizons:

$$\mathcal{L} = \frac{1}{N \cdot H} \sum_{i=1}^{N} \sum_{j=1}^{H} (y_{i,j} - \hat{y}_{i,j})^2$$

**Optimiser and regularisation.** Adam optimiser (learning rate 0.001), gradient clipping ($\|\nabla\|_{\max} = 1.0$), dropout ($p = 0.3$), weight decay ($\lambda = 10^{-5}$), early stopping (patience 8 on validation loss), and ReduceLROnPlateau (factor 0.5, patience 3). Batch size 32; maximum 50 epochs; input window 12; horizon 6.

**Multi-seed design.** Each of the four architectures was trained across **twenty independent seeds (42–61)** for a total of 80 training runs. All hyperparameters, windowing, data split, and optimisation settings were identical across architectures and seeds; only the architecture (and the seed for stochastic initialisation/data ordering) varied. This isolates the architectural effect from confounds of optimisation and stochastic initialisation. We note that this design deliberately holds hyperparameters fixed across architectures rather than tuning each architecture separately; per-architecture hyperparameter optimisation could in principle shift the ranking, but such tuning would trade away the controlled, attribution-clean comparison that is the primary goal of a robustness audit (and, if performed without a held-out protocol, introduces its own risk of overfitting).

**Data preprocessing.** Z-score normalisation (mean/SD computed from training set only), temporal encoding (hour/minute normalised to a [0,1] range), and sliding-window generation with stride 1.

**Hardware.** All models were trained on a 2-core CPU; on the full T1D cohort each configuration converged within a mean of 70–83 min (seed range 42–186 min) and within 9–10 epochs.

### 2.5 Statistical Analysis

Four architectures × twenty seeds produce per-architecture, per-seed RMSE values, enabling pairwise comparisons (6 pairs). Because the between-architecture RMSE differences are small relative to within-architecture seed variability, our primary inference is based on **effect size against a clinically meaningful threshold**, with secondary reporting of significance tests. As the glucose-forecasting literature does not currently provide a single, universally accepted numerical value for a clinically meaningful *between-model* RMSE difference (reported benchmarks span a wide range of mean RMSE across horizons and cohorts [14]), we adopt a priori an **operational anchor** of 5–10 mg/dL, derived transparently rather than quoted as a fixed clinical standard. The anchor is constructed from two independent, defensible inputs. First, the internationally recognised CGM accuracy consensus requires clinically acceptable systems to achieve a mean absolute relative difference (MARD) below 10% and Clarke Error-Grid A+B rates above 98% [2, 4]; this is a genuine consensus on *device/forecast accuracy*, and it sets the scale of error that clinical practice already treats as acceptable. Second, our own MARD results (≈7.4%, §3.3.2) provide an empirically estimated per-window relative-error floor consistent with that consensus. A between-model RMSE difference of 5–10 mg/dL corresponds to a roughly 20–60% relative change at the operating RMSE observed here (~16–23 mg/dL across cohorts); as point(s) of reference, this is of the same order as, or larger than, the spread in mean RMSE between leading published models at comparable horizons [14], so it is not an arbitrarily small bar. We emphasise that this is an **interpretive and operational benchmark** for comparing effect sizes, not a claim that a 5–10 mg/dL difference is catastrophic in clinical practice; our substantive conclusion rests jointly on this distance-to-threshold and on the independent multi-metric clinical assessment (§2.6), in which the statistically distinct architectures are clinically indistinguishable, rather than on the threshold alone. Conversely, the between-architecture differences we observe (0.045–0.28 mg/dL) are **≈18–111-fold smaller** than this anchor. The threshold is used solely as an interpretative benchmark for comparing effect sizes, and our conclusions are driven by this distance-to-threshold alongside the multi-metric clinical assessment (§2.6), rather than by the threshold alone. The following analyses were applied:

| Analysis | Purpose |
|:-----|:--------|
| **Paired t-test** | RMSE mean difference (paired seeds) |
| **Wilcoxon signed-rank** | Robust non-parametric confirmation |
| **Cohen's d** | Paired effect size (|d| ≈ 0.2 small, 0.5 medium, 0.8 large) |
| **Holm–Bonferroni correction** | Multiple-comparison correction (6 pairs) |
| **Bootstrap (1,000×)** | 95% CI of RMSE difference, dedicated RNG (seed 20260820) |
| **Diebold–Mariano (DM)** [21] | Confirmatory comparison of forecast RMSE differences at the seed level (reported alongside the primary paired t-test) |

> **Note on power.** Because the paired comparisons use the same subject-level data split across seeds, these tests characterise sensitivity to stochastic initialisation and data-ordering under a fixed split, not uncertainty due to re-sampling the subject population; the leave-one-subject-out alternative is discussed in §4.5. With n = 20 seeds (df = 19), statistical power is substantially higher than the 5-seed designs common in the literature, allowing reliable inference on between-architecture differences. We therefore interpret the conjunction of: (i) at most two pairwise comparisons remaining significant after Holm–Bonferroni correction under either the paired t-test (Holm p = 0.041) or the confirmatory Diebold–Mariano metric (Holm p = 0.014 and 0.022), all at magnitudes ≈77–111-fold below clinical relevance; (ii) standardised effect sizes (Cohen's d = 0.16–0.68, medium by convention but corresponding to raw RMSE differences of only 0.012–0.065 mg/dL); and (iii) between-architecture RMSE differences ≈77-fold (well over an order of magnitude) below the a priori 5–10 mg/dL clinically meaningful threshold — as evidence that **no clinically meaningful difference exists**, rather than as a failure to detect one.

### 2.6 Clinical Evaluation

Beyond regression metrics, we assessed clinical applicability on the test set (219 subjects; 219,000 windows flattened to 1,314,000 prediction–actual pairs at the original mg/dL scale), using the seed-42 models. Acknowledging recent critiques that raw RMSE can bias evaluation toward trivial target-range predictions and therefore requires complementary clinical criteria [27], we report a multi-metric clinical assessment alongside the regression-level comparisons:

- **Clarke error-grid analysis (EGA)** [22], reported at two resolutions with two distinct definitions. (i) **Standard Clarke EGA** (full A–E zone grid) reported pooled across all six horizons; (ii) **±20% relative-error accuracy** (a simplified relative-error criterion, distinct from the full Clarke grid: a prediction is scored clinically acceptable when it lies within ±20% of the reference, a slightly more permissive threshold than the ISO 15197:2013 ±15% / ±15 mg/dL requirement for point-of-care glucose accuracy [28], adopted here as a conservative tolerance for forecasted values), reported per-horizon. We explicitly distinguish these two metrics to avoid conflating them.
- **Mean absolute relative difference (MARD)**, disaggregated by architecture (20-seed mean ± SD).
- **Stratified RMSE** by blood-glucose range: hypoglycaemia (<70), euglycaemia (70–180), hyperglycaemia (>180 mg/dL).
- **Per-horizon RMSE** at 5, 10, 15, 20, 25, and 30 minutes.
- **Hypoglycaemia detection** (<70 mg/dL): sensitivity, specificity, positive and negative predictive values.

Reproducibility is ensured via released scripts (`run_clinical_inference.py`, `clinical_evaluation.py`) and a fixed random seed (42) for clinical prediction generation.

> **Note on clinical-evaluation seed.** Consistent with standard practice for clinical error analysis, per-window clinical metrics (Clarke EGA, per-horizon accuracy, stratified and per-horizon RMSE, hypoglycaemia detection) were computed on the seed-42 models of each architecture. The seed-42 test RMSE values (LSTM 16.261 / BiLSTM 16.182 / LSTM-Attention 16.244 / BiLSTM-Attention 16.202 mg/dL) lie within 0.4% of the corresponding 20-seed means (16.296±0.074 / 16.243±0.054 / 16.276±0.059 / 16.231±0.055 mg/dL), so the clinical conclusions, which are driven by effect sizes far below clinical thresholds rather than by any single-seed ranking, are robust to the choice of reporting seed. MARD, which depends more strongly on the per-window error distribution, is reported as a 20-seed mean ± SD.

---

## 3 Results

### 3.1 Model Performance: Architecture Complexity Yields Negligible Gains

Table 1 summarises the 20-seed mean ± SD performance of the four architectures on the MetaboNet test set at the 30-minute horizon, and **Figure 1** shows each architecture's per-seed RMSE dispersion for both the primary T1D cohort and the T2D validation cohort, demonstrating that between-architecture differences within a cohort are of the same order as, or smaller than, between-seed variation and that all architectures degrade by a consistent, small margin when transferred to the harder T2D cohort.

**Table 1. Model performance comparison on the MetaboNet test set at the 30-minute prediction horizon (20-seed mean ± SD; 219 independent test subjects / 219,000 windows). Train time is the mean over 20 seeds.**

| Model | Parameters | RMSE (mg/dL) | MAE (mg/dL) | R² | Train time (min) |
|:------|:----------:|:------------:|:-----------:|:--:|:----------------:|
| LSTM | 208,838 | 16.296 ± 0.074 | 9.958 ± 0.162 | 0.9236 ± 0.0007 | 70.1 |
| BiLSTM | 143,302 | 16.243 ± 0.054 | 9.844 ± 0.092 | 0.9240 ± 0.0005 | 71.5 |
| LSTM-Attention | 212,998 | 16.276 ± 0.059 | 9.900 ± 0.094 | 0.9237 ± 0.0006 | 74.5 |
| BiLSTM-Attention | 147,462 | 16.231 ± 0.055 | 9.840 ± 0.095 | 0.9242 ± 0.0005 | 82.8 |

**Key observations:**

1. **Narrow performance band.** All four architectures achieved mean RMSE within 16.23–16.30 mg/dL and R² within 0.9236–0.9242. The largest between-architecture RMSE difference — LSTM (worst) vs. BiLSTM-Attention (best), 0.065 mg/dL — is far below the clinically meaningful threshold for sensor-grade accuracy (≥5–10 mg/dL, our a priori criterion defined in §2.5).
2. **Plain LSTM is sufficient.** The plain LSTM (RMSE 16.296) matched the most complex BiLSTM-Attention (16.231) within 0.065 mg/dL (~0.40% relative), despite having *more* parameters.
3. **Strong gains over non-learning baselines.** All deep models improved on persistence (RMSE 27.37 mg/dL) by ≈40.5–40.7% and on linear extrapolation (26.64 mg/dL) by ≈38.8–39.1%, demonstrating that learned models capture glycaemic dynamics beyond short-term autocorrelation (Table 1).
4. **No spurious winner from single seeds.** Within-architecture seed variation (RMSE SD 0.054–0.074 mg/dL) was of the same order as, or larger than, between-architecture differences (**Figure 2**). Had any single seed been used for model selection, one of the four architectures could have been spuriously ranked best; the pairwise Diebold–Mariano tests in **Figure 2** show that at most two contrasts (LSTM vs. BiLSTM-Attention and LSTM-Attention vs. BiLSTM-Attention) are statistically significant after Holm correction, both at magnitudes far below clinical relevance.

### 3.2 Statistical Analysis: No Clinically Meaningful Architectural Difference

Table 2 reports pairwise comparisons of RMSE across architectures (n = 20 seeds).

**Table 2. Pairwise statistical comparison of RMSE across architectures (n = 20 seeds). ΔRMSE = RMSE(A) − RMSE(B); positive favours B (right-listed model). Effect size Cohen's d (pairwise dz); raw and Holm–Bonferroni-corrected p-values for the paired t-test, Wilcoxon signed-rank, and a seed-level Diebold–Mariano-type [21] comparison of per-seed RMSE differences; bootstrap 95% CI of the RMSE difference (1,000 resamples, fixed RNG seed 20260820).**

| Comparison (A vs B) | ΔRMSE | Cohen's d | t-test p | Wilcoxon p | DM p | Holm (t) | Holm (W) | Holm (DM) | Bootstrap 95% CI | Conclusion |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----|
| LSTM vs BiLSTM | +0.053 | +0.53 | 0.028 | 0.017 | 0.017 | 0.111 | 0.069 | 0.068 | [+0.011, +0.097] | n.s. |
| LSTM vs LSTM-Attention | +0.019 | +0.22 | 0.346 | 0.522 | 0.334 | 0.692 | 1.000 | 0.667 | [−0.019, +0.061] | n.s. |
| LSTM vs BiLSTM-Attention | +0.065 | +0.68 | 0.007 | 0.006 | 0.002 | **0.041** | **0.034** | **0.014** | [+0.024, +0.106] | **sig.** |
| BiLSTM vs LSTM-Attention | −0.033 | −0.44 | 0.066 | 0.090 | 0.051 | 0.198 | 0.269 | 0.153 | [−0.067, −0.0003] | n.s. |
| BiLSTM vs BiLSTM-Attention | +0.012 | +0.16 | 0.490 | 0.571 | 0.482 | 0.692 | 1.000 | 0.667 | [−0.020, +0.043] | n.s. |
| LSTM-Attention vs BiLSTM-Attention | +0.045 | +0.64 | 0.010 | 0.014 | 0.004 | 0.051 | 0.068 | **0.022** | [+0.016, +0.076] | n.s. (DM sig.) |

† *“t-test p”, “Wilcoxon p”, and “DM p” are the raw paired t-test, Wilcoxon signed-rank, and seed-level Diebold–Mariano-type [21] p-values; “Holm (t)”, “Holm (W)”, and “Holm (DM)” are the corresponding Holm–Bonferroni-corrected values (six contrasts per family). The outcome is robust across tests: only LSTM vs. BiLSTM-Attention is significant after Holm correction (t-test 0.041; Wilcoxon 0.034; DM 0.014), with one additional DM-significant contrast (LSTM-Attention vs. BiLSTM-Attention, Holm DM 0.022). All significant effects are 0.045–0.065 mg/dL, ≈77–111-fold below the a priori clinical threshold (5 mg/dL) — statistically detectable yet clinically negligible. Bootstrap 95% CIs exclude zero for four of six contrasts at the raw level; all magnitudes remain far below clinical relevance. For BiLSTM vs. LSTM-Attention the bootstrap 95% CI ([−0.067, −0.0003]) marginally excludes zero whereas the paired t-test is not significant (p = 0.066); this reflects the different construction of the two intervals (percentile bootstrap of the per-seed difference vs. a normal-theory test on the mean), and both indicate a negligible magnitude (0.033 mg/dL) far below clinical relevance.*

**Honest conclusion.** The central evidence for diminishing architectural returns rests on the **effect size and clinical meaningfulness**, not on p-values: the largest between-architecture RMSE difference was **0.065 mg/dL**, ≈77-fold (well over an order of magnitude) below the 5–10 mg/dL threshold we adopt a priori as clinically meaningful for sensor-grade accuracy (§2.5). Across the six pairwise comparisons, at most two contrasts (LSTM vs. BiLSTM-Attention, DM Holm p = 0.014 / t-test Holm p = 0.041; LSTM-Attention vs. BiLSTM-Attention, DM Holm p = 0.022) remained significant after Holm–Bonferroni correction, and these correspond to just 0.045–0.065 mg/dL, i.e., they are **statistically detectable but clinically negligible**. Because this study used n = 20 seeds (df = 19) — substantially greater statistical power than the 5-seed designs common in the literature — the combination of at most two nominally-significant-but-clinically-negligible contrasts, standardised effect sizes of medium magnitude whose raw RMSE differences (≤0.065 mg/dL) stay far below the clinical threshold, and between-architecture RMSE differences ≈77-fold (well over an order of magnitude) below the a priori clinically meaningful threshold constitutes strong evidence that any real architectural differences are too small to be clinically meaningful or to support architecture selection at this horizon. The observed ranking by point estimate (BiLSTM-Attention 16.231 < LSTM 16.296, with BiLSTM 16.243 and LSTM-Attention 16.276 in between) is descriptive only and driven by noise; no consistent winner emerged across seeds.

**T2D validation.** The same honest reading applies to the independent T2D validation cohort (Figures 1–3). Across the six pairwise comparisons in T2D, three contrasts remained significant after Holm–Bonferroni correction of the Diebold–Mariano tests (LSTM vs. BiLSTM, DM p < 0.001; BiLSTM vs. LSTM-Attention, DM p = 0.014; LSTM vs. BiLSTM-Attention, DM p = 0.042). Their magnitudes (0.18–0.28 mg/dL) are larger than the corresponding T1D effects but remain ≈18–28-fold below the a priori 5 mg/dL clinical threshold, so the substantive conclusion — added architectural complexity yields no clinically meaningful gain — persists across both populations, and the observed T2D differences again fall within the same order of magnitude as the between-seed dispersion of the 20-seed audit.

### 3.3 Clinical Evaluation: Short-Horizon Acceptability, Long-Horizon and Hypoglycaemia Limitations

#### 3.3.1 Clinical Acceptability: Standard Clarke EGA and Per-Horizon ±20% Accuracy

Clinical evaluation of the BiLSTM-Attention model (reported as the representative architecture, with per-architecture metrics given below) and all architectures was quantified directly from per-window errors, and is summarised in **Figure 5**. **Standard Clarke EGA** at the 30-minute horizon gave pooled Zone A of 77.8–79.3% with a negligible fraction (≤0.0024%; 16–31 of 1,314,000 samples, 0.0012–0.0024%) in dangerous Zone E, and a Bland–Altman analysis of the per-window error indicated a small mean bias with narrow limits of agreement. **Per-horizon ±20% relative-error accuracy** (a distinct, conventional criterion for short-horizon assessment) was high and all architectures met the clinically acceptable >95% level at horizons ≤ 15 minutes:

| Horizon | LSTM | BiLSTM | LSTM-Attention | BiLSTM-Attention |
|:--------|:----:|:------:|:--------------:|:----------------:|
| 5 min | 99.7% | 99.7% | 99.7% | **99.7%** |
| 10 min | 98.8% | 98.8% | 98.9% | **98.9%** |
| 15 min | 96.6% | 96.4% | 96.7% | **96.7%** |
| 20 min | 93.3% | 92.8% | 93.4% | **93.4%** |
| 25 min | 89.6% | 88.8% | 89.6% | **89.6%** |
| 30 min | 85.8% | 84.9% | 85.7% | **85.8%** |

All architectures met the clinically acceptable >95% ±20% criterion at ≤ 15-minute horizons. **We report the standard Clarke EGA (77.8–79.3% pooled Zone A) and the per-horizon ±20% criterion separately**, and explicitly identify horizons > 15 minutes as a limitation rather than claiming a pooled >95% clinical acceptability.

#### 3.3.2 MARD and Per-Horizon RMSE

MARD was ≈ 7.33–7.43% across architectures (LSTM 7.43 ± 0.18%; BiLSTM 7.34 ± 0.13%; LSTM-Attention 7.37 ± 0.13%; BiLSTM-Attention 7.33 ± 0.12%) — comfortably below the <10% CGM clinical standard. Per-horizon RMSE grew linearly with horizon (≈5.8 mg/dL at 5 min → ≈24 mg/dL at 30 min), with BiLSTM-Attention optimal or tied at nearly all horizons.

#### 3.3.3 Stratified Error

RMSE followed a U-shape across glucose ranges: lowest in euglycaemia (70–180 mg/dL, ≈13.5–14.1), higher at both extremes (hypoglycaemia <70: ≈21.9–24.4; hyperglycaemia >180: ≈20.9–21.8 mg/dL). This reflects the physics of glucose dynamics — wider fluctuations near the boundaries.

#### 3.3.4 Hypoglycaemia Detection

For hypoglycaemia (<70 mg/dL; 3.02% of test samples), all models showed **high specificity (>99.5%) and good PPV (74–86%) but moderate sensitivity (32–47%)** — i.e., ≈53–68% of hypoglycaemic events were missed, a clinically important limitation. BiLSTM had the lowest sensitivity (32.75%) but highest PPV (86.15%), reflecting a more conservative (miss-not-err) operating point. Class imbalance (low-hypoglycaemia prevalence) and higher error near low thresholds likely contribute [19]; we identify hypoglycaemia sensitivity as the primary clinical improvement direction.

### 3.4 Comparison with Prior Work

Consistent with prior deep-learning studies on the OhioT1DM cohort (12 subjects; 30-min RMSE ≈17.5–18.7 mg/dL [7, 9, 13]), the slightly lower absolute RMSE (16.23–16.30 mg/dL) we observe is encouraging given MetaboNet's comparable sensor-grade accuracy at far greater subject-count heterogeneity (1,092 vs. 12). Notably, **our mean RMSE is lower in absolute terms than several smaller-cohort studies**, and our mean R² (≈0.92) is at the favourable end of the ≈0.80–0.90 range typically reported for real-world deep-learning BG forecasting [7, 9, 10].

Crucially, by fixing the training pipeline and varying only architecture across twenty seeds, we isolate the architectural effect — a design most prior CGM comparisons (single seed, non-uniform hyperparameters) cannot support [10, 13]. The "attention advantage" reported in several single-seed studies largely disappears under controlled, multi-seed comparison, echoing a broader pattern across time-series forecasting where sophisticated models frequently fail to beat simple baselines under rigorous evaluation [20].

We further benchmark against the MetaboNet dataset's own reference forecasting results reported in the dataset paper [15]: the lightweight non-learning and classical baselines published with the cohort (e.g., zero-order hold and linear extrapolation, reported at comparable horizons) bracket the accuracy achievable without learned extrapolation, and our deep models' improvements over the persistence baseline (27.37 mg/dL, −40.5–40.7%) place the learned approaches clearly above these reference points while confirming that, within the deep-learning family, added complexity is not rewarded.---


## 4 Discussion

### 4.1 Principal Findings

Using the MetaboNet dataset — the largest publicly available consolidated set of real-world, multi-centre continuous glucose monitoring (CGM) traces from people with type 1 diabetes (T1D) [15] — we conducted a systematic multi-seed robustness-benchmarking study comparing four recurrent architectures of increasing nominal complexity: a plain LSTM, a bidirectional LSTM (BiLSTM), an LSTM augmented with an additive attention module (LSTM-Attention), and a BiLSTM-Attention hybrid [7, 8, 10, 11]. Across 1,092 individuals (219 independent test subjects; 219,000 test windows), evaluated at a 30-minute prediction horizon, all four architectures performed remarkably similarly, with mean RMSE values confined to a narrow band of **16.23–16.30 mg/dL**, mean MAE of **9.84–9.96 mg/dL**, and mean R² of **0.9236–0.9242** (Table 1). The largest difference between any two architectures was **0.065 mg/dL** in RMSE.

Three findings merit emphasis. **First**, the aggregate benefit of architectural complexity was negligible (**Figure 3**): the difference between the plain LSTM and the most complex BiLSTM-Attention was only 0.065 mg/dL (~0.40% relative), far below any clinically meaningful threshold (typically ≥5–10 mg/dL for sensor-grade accuracy, our a priori criterion defined in §2.5). **Second**, this performance was achieved against strong non-learning baselines: all deep models improved on persistence (RMSE 27.37 mg/dL) by ≈40.5–40.7% and on linear extrapolation (26.64 mg/dL) by ≈38.8–39.1%. **Third**, the multi-seed audit revealed that within-architecture seed variation (RMSE standard deviation 0.054–0.074 mg/dL across twenty seeds) was of the same order as, or larger than, the between-architecture differences. This last observation is central: had any single seed been used as the basis for model selection, one of the four architectures could have been spuriously ranked best, yet no consistent winner emerged. Our results therefore refute the implicit assumption that more sophisticated recurrent architectures yield meaningful predictive gains on univariate, CGM-only glucose forecasting, and demonstrate the value of multi-seed robustness auditing in medical AI benchmarking [17, 20].

### 4.2 Comparison with Prior Work

Our results are consistent with, and extend, the emerging evidence that gains from architectural complexity in blood glucose (BG) prediction are modest. Prior deep-learning studies on the OhioT1DM cohort (12 subjects) reported 30-minute RMSE values of ≈17.5–18.7 mg/dL for CNN-RNN and CNN-LSTM models [7, 9], and transformer-based approaches have reported 15–18 mg/dL on the same small cohort [13]. The slightly lower absolute RMSE (16.23–16.30 mg/dL) we observe is informative: MetaboNet aggregates far more subjects (1,092 vs. 12) and correspondingly greater phenotypic, sensor, and centre-level heterogeneity [15], yet our multi-seed evaluation attains comparable-or-better accuracy, so a directly comparable single-site cohort should be interpreted alongside this broader benchmark rather than as a ceiling. Within the R² range typically reported for deep-learning BG forecasting on real-world data (≈0.80–0.90 at 30 min) [7, 9, 10], our mean R² of ≈0.92 is at the favourable end.

We further align our evaluation with recent calls in the glucose-forecasting community to move beyond raw accuracy toward clinically relevant performance criteria [27]. By reporting, alongside RMSE, standard Clarke error-grid analysis, MARD against the <10% consensus threshold [2, 4], ±20% per-horizon accuracy, stratified error, and hypoglycaemia detection (§2.6, §3.3), we directly address the concern that accuracy metrics alone can favour trivial target-range predictions and obscure clinically relevant failures. This multi-metric clinical reading substantiates our central claim on the primary T1D cohort: the architecture pairs that differ statistically (LSTM vs. BiLSTM-Attention, ΔRMSE 0.065 mg/dL) are clinically indistinguishable on MARD (7.33–7.43%), Clarke-Zone distribution (77.8–79.3% Zone A; ≤0.0024% Zone E), and hypoglycaemia sensitivity (0.33–0.47) — so the absence of a consequential accuracy difference is corroborated across independent clinical lenses rather than resting on the RMSE threshold alone. (Clinical metrics were computed on the primary T1D cohort; the T2D cohort served as a subject-disjoint validation of the statistical pattern, as described in §2.1 and §3.2.)

Crucially, our design goes beyond earlier single-model comparisons. Because we held the training pipeline, windowing, and all hyperparameters fixed and varied **only** the architecture, and repeated each configuration across twenty seeds, we isolate the architectural effect from confounds of optimisation and stochastic initialisation. Previous comparative studies in this domain, including those that report attention-based gains [10, 13], frequently evaluate a single random seed or non-uniform hyperparameters, making it impossible to determine whether reported advantages reflect the architecture or incidental training conditions. Our finding that the previously reported "attention advantage" largely disappears under controlled, multi-seed comparison echoes a broader pattern documented across time-series forecasting, where sophisticated models frequently fail to beat simple baselines under rigorous evaluation [20].

### 4.3 Generalisability and Methodological Implications

#### 4.3.1 Extending the multi-seed robustness-benchmarking framework to other medical time-series tasks

The methodological contribution of this study — a controlled, multi-seed robustness-benchmarking protocol that quantifies the *marginal* value of architecture complexity subject to a fixed training protocol — is directly transferable to a broad class of medical time-series prediction problems that share the structure of sequential, moderately autocorrelated physiological signals. Obvious targets include ICU vital-sign forecasting (heart rate, blood pressure, respiratory rate), where the information content of short sequences is bounded by physiological inertia and measurement noise rather than by model capacity; ECG rhythm classification and QT-interval trend prediction [11]; and respiratory-rate and oxygen-saturation trend monitoring in acute care. In all of these settings, the same question arises that we answer here for glucose: *does a heavier architecture add meaningful accuracy, or is the performance ceiling set by the signal itself?* The framework we propose — fix the optimisation pipeline, vary architecture only, repeat across multiple seeds, and report between-seed dispersion alongside point estimates — provides a principled template for answering that question fairly [17]. We encourage the community to adopt this protocol for benchmarking foundation models and transformer-based architectures proposed for physiological data [12].

#### 4.3.2 Is diminishing architectural returns a general phenomenon in medical AI?

Our empirical result sits within a growing body of evidence indicating that, for many medical forecasting and classification tasks, the marginal return on model complexity is small or absent once a reasonable recurrent or convolutional baseline is reached. For CGM glucose prediction specifically, Mirshekarian et al. found LSTMs and attention-augmented variants achieved comparable performance, with the benefit of attention lying more in interpretability than in accuracy [10]; our results are consistent with and substantially extend this on a cohort that is ~90-fold larger. Beyond glucose, comparable "simpler is sufficient" findings have been reported in agricultural and financial time-series forecasting, where sophisticated gradient-boosting or deep models frequently fail to achieve statistically significant gains over simple statistical benchmarks under statistically valid evaluation, and in short-horizon physiological forecasting where baseline and linear models remain competitive [20].

Theoretically, there are compelling reasons why medical time-series may be insensitive to architectural complexity, and these reasons generalise beyond glucose. First, physiological signals are bounded, reflect strongly autocorrelated underlying homeostatic processes, and are corrupted by sensor noise, irregular sampling (gaps, artefacts, sensor dropouts), and inter-subject and intra-subject variability — all forms of noise that no network capacity can remove. Second, for short forecasting horizons (e.g., ≤30–60 min), the dominant source of forecast skill is the recent trend and the inertia of the process itself; a recurrent unit of even moderate capacity trivially captures this, so additional parameters act on a component that is already explained. Third, and most fundamentally, attainable accuracy is limited not by model capacity but by irreducible aleatoric uncertainty — the fact that future glucose (or heart rate, or respiration) depends on unobserved inputs such as meals, stress, physical activity, illness, and medication. No amount of architectural sophistication operating on a univariate input channel can recover information that is simply not present in that channel [15]. In information-theoretic terms, the predictively available signal is a property of the *data generating process* and the *input channel*, not of the model family; a more expressive model can at best approximate the same Bayes-optimal predictor, and with finite, noisy data it will usually do so no better — and sometimes worse — than a simpler one. This is why our twenty-seed audit matters: a single architecture compared at a single seed can spuriously appear superior merely by chance, and it is precisely this sampling artefact that multi-seed repetition suppresses. Consequently, for univariate physiological forecasting over short horizons, we should expect — and now empirically confirm — diminishing returns from complexity. This argues for re-allocating research effort from ever-larger architectures toward (i) richer input modalities, (ii) uncertainty quantification, and (iii) rigorous, reproducible benchmarking. 

#### 4.3.3 Implications for methodology in medical AI benchmarking

Our study underscores three methodological recommendations. First, **baseline discipline**: no deep-learning model should be claimed to add value without head-to-head comparison against both a strong non-learning baseline (persistence, linear extrapolation) and a simple recurrent baseline, under an identical and leakage-free evaluation protocol — a principle echoed by recent critiques of deep time-series claims [20]. Second, **multi-seed robustness auditing**: because between-seed dispersion here routinely exceeded between-architecture differences, single-seed comparisons are statistically uninformative for model ranking; we recommend reporting mean ± SD over ≥5 seeds for every reported metric. Third, **effect-size awareness**: statistical significance is insufficient; researchers should interpret differences against clinically meaningful thresholds (e.g., the 5–10 mg/dL sensor-accuracy criterion we adopt a priori in §2.5) to avoid over-claiming tiny but "significant" gains. A simple rule of thumb follows from our data: if the between-architecture difference is smaller than the within-architecture seed spread, the difference is not empirically resolvable and should not drive model choice. Together, these practices would substantially reduce the reproducibility crisis and the "architecture treadmill" that pervades medical AI publication.

### 4.4 Clinical Implications

#### 4.4.1 Implications for T1D self-management and hypoglycaemia prevention

CGM-based glucose forecasting underpins several clinically deployed diabetes technologies, most notably predictive low-glucose suspend (PLGS) systems, which suspend insulin delivery when glucose is predicted to cross a threshold, and advanced hybrid closed-loop (AHCL) systems that modulate basal insulin and deliver correction boluses on the basis of predicted glucose [3, 4]. In these systems, the forecast is executed on-device and in real time, within strict latency and power budgets. Our finding that a plain LSTM matches a substantially more complex BiLSTM-Attention architecture (within 0.065 mg/dL at the 30-minute horizon) has direct, concrete implications for this context. PLGS and AHCL algorithms need only enough predictive accuracy to reliably flag imminent hypoglycaemia and rise/fall trends; the extra accuracy obtainable from attention mechanisms is orders of magnitude below the resolution of even the most rigorous clinical thresholds (time below range, TBR <70 mg/dL; time in range, TIR 70–180 mg/dL) [4]. A simpler LSTM therefore appears *sufficient* for the real-time forecasting component of such systems, provided it is embedded within a validated safety architecture. This is not a claim that forecasting complexity is irrelevant at longer horizons or for other tasks; it is a scoped, evidence-based statement for the 30-minute, CGM-only use case we evaluated.

#### 4.4.2 Computational and deployment implications across resource settings

The computational profile of the models reinforces their clinical deployability (**Figure 4**). On the full T1D cohort, all four architectures train in about 70–83 min on 2-core CPU hardware; the same architectures train roughly ten times faster (mean ~7.5 min; range 4.8–10.9 min across the 80 runs) on the smaller T2D validation cohort, and all converge within 9–10 epochs with modest parameter counts (≈140–213 K) (Table 1). Such lightweight models are well suited to on-pump and on-sensor inference and to mobile/edge deployment, where memory footprint and inference latency are binding constraints — an active and important area given the push towards fully automated, battery-powered closed-loop devices [3]. Conversely, the more complex architectures take *no less* time and *no fewer* parameters while delivering no accuracy benefit; in fact, the simpler plain LSTM has *more* trainable parameters (208,838) than the BiLSTM-Attention (147,462), yet offers no advantage — and, crucially, the lightest models (BiLSTM, 143,302 parameters) match the heaviest. This is an important practical nuance: because the plain LSTM is the simplest model to implement and deploy and is one of the lighter-configuration options we tested, its performance parity means the community is not sacrificing anything clinically meaningful by choosing the simpler, cheaper option. Clinically, lower computational burden translates into longer battery life, faster wake-up times for periodic forecasting, and easier software certification and on-device updating — all operational constraints that shape whether a predictive feature can be shipped in a regulated medical device at all.

For resource-constrained settings — primary-care and district-level hospitals, and low- and middle-income countries where advanced computing hardware and reliable connectivity are limited and where diabetes burden is disproportionately high [1] — a validated LSTM provides an accessible, low-power, easily maintained baseline that clinicians and engineers can deploy and update without specialised infrastructure. In telemedicine and smartphone-GUI-based self-management, a lightweight model can be converted to formats such as TensorFlow Lite and run locally on consumer devices, preserving data privacy and functioning offline [3]. This aligns with a broader trend in medical AI toward lightweight models that preserve accuracy while enabling edge deployment for ECG (arrhythmia) and other wearable applications [18].

#### 4.4.3 Guiding clinical model selection and the industry view

For clinicians and technology evaluators, our results argue that model selection should rest on *measurable clinical endpoint performance* (forecast accuracy against persistent/baseline references, real-time latency, demonstrated safety in use) rather than on architectural novelty. The "architecture complexity" of a forecasting model is, on the present evidence, not a reliable proxy for clinical utility at the 30-minute CGM horizon. For the diabetes-technology industry, this offers a practical and economic signal: CGM sensor and pump manufacturers need not incur the engineering cost, validation burden, and regulatory overhead of deploying the most architecturally complex models to achieve state-of-the-art *univariate* short-horizon forecasting. In regulated medical-device development, every added model complexity carries a real cost — longer verification & validation cycles, larger safety-case documentation for regulatory bodies, higher memory and power budgets on hardware, and greater maintenance burden across software updates. Demonstrating that a simpler architecture attains parity therefore shortens time-to-market and lowers the barrier to entry for smaller developers and open-source implementations, which can in turn increase market competition and device accessibility [3]. A simpler, reproducible, and open-source LSTM can serve as a strong production baseline, freeing engineering resources for the aspects that demonstrably matter — multimodal sensor fusion, robust uncertainty estimation, and rigorous prospective clinical validation. The public release of our reproducible pipeline (data loading, model implementations, and evaluation scripts) supports such industry and academic reuse.

### 4.5 Limitations

We are explicit about the boundaries of our claims.

**Dataset limitations.** (i) The primary MetaboNet cohort comprises solely T1D participants. We addressed cross-population transfer by validating all four architectures on an independent T2D cohort (ShanghaiT2DM, n = 100; §2.1), in which the primary conclusions — no clinically meaningful architectural gain and between-seed variation dominating between-architecture differences — were reproduced (Figures 1–3). This T2D validation cohort is small (n = 100) and single-site, though it is fully subject-disjoint (strict 60/20/20 patient split with zero overlap, §2.1); extension to larger, multi-centre T2D cohorts remains warranted. (ii) The models were trained and evaluated on **CGM alone**, without insulin dosing, carbohydrate intake, physical activity, or stress information; multimodal inputs have been shown to improve accuracy and are required for longer-horizon prediction [15]. (iii) The aggregated, multi-centre composition and the specific sensor generations represented in MetaboNet [15] mean that our results benchmark the *current*-generation univariate CGM signal; novel sensor technology or denser sampling could in principle alter the complexity–accuracy trade-off, an empirical question we do not presume to answer.

**Methodological limitations.** (i) We evaluated only four recurrent architectures; we did not include temporal convolutional networks (TCNs), pure transformer encoders, state-space (Mamba-family) models, or the increasingly popular "simple MLP" forecaster baselines [20], and we restricted attention to a single 30-minute horizon. It is therefore possible that some architectures we did not test, or longer horizons (60/120 min) where temporal dependencies are more complex, could exhibit non-trivial accuracy differences — indeed, longer-horizon forecasting is where architecture choice is most likely to matter, and our "diminishing returns" conclusion is explicitly scoped to the 30-minute, univariate setting. (ii) Cross-population generalisability of the *quantitative* benchmark was assessed on a single small T2D cohort (§2.1); we did not include OhioT1DM or other independent external cohorts, so the quantitative headroom across further populations and sensor generations remains to be confirmed, even though the qualitative pattern (small architectural differences persisting across T1D and T2D) is internally consistent. (iii) The evaluation uses a subject-based split; we did not perform leave-one-subject-out cross-validation across all 1,092 individuals, and we did not report calibration or prediction intervals, which would be required for probabilistic clinical use.

**Clinical-translation limitations.** (i) We validated predictive accuracy only; we did not deploy the models within an actual closed-loop pump system or a PLGS algorithm, and we make no claim regarding in-silico or in-vivo safety, time-in-range outcomes, or hypoglycaemia-event reduction. (ii) We pooled all subjects and did not model patient-specific factors — age, diabetes duration, residual beta-cell function, comorbidities, or sensor site/type — that previous work shows can materially alter per-patient forecasting and personalisation benefits [10, 15]. (iii) The mean per-configuration CPU training time of ~70–83 min (T1D) and ~7.5 min (T2D; range 4.8–10.9 min) with 9–10 epochs were incurred in a research setting; production on-device training and latency were not measured. These limitations bound the clinical recommendations we can responsibly offer and define the scope of future work.

### 4.6 Future Directions

Our findings motivate several concrete research and translation directions.

1. **Broader cross-population validation.** Having reproduced the central findings on an independent small T2D cohort (n = 100), the multi-seed robustness-benchmarking protocol should next be applied to larger, multi-centre T2D cohorts and to OhioT1DM [23] and other sensor generations, to solidify quantitative generalisability rather than the qualitative consistency demonstrated here.

2. **Multimodal and longer-horizon forecasting.** Because short-horizon univariate skill is bounded by the information present in the glucose channel, the most promising route to accuracy gains is adding insulin, meal, activity, and stress modalities [15], and extending to 60-min, 120-min, and overnight horizons [16] — precisely the regime in which architectural complexity, and attention and state-space mechanisms, may begin to pay off.

3. **Embedded/on-device deployment.** Quantifying real-time inference latency, memory footprint, and power consumption of each architecture on representative on-pump/mobile hardware would translate our accuracy findings into concrete deployment guidance for PLGS and AHCL systems and resource-constrained settings [3, 18].

4. **Personalised and patient-specific models.** The intentional pooling of subjects in this study motivates follow-up with per-patient fine-tuning, few-shot personalisation, and meta-learning approaches [10] to determine whether personalisation — rather than architectural complexity — is the dominant lever for individual-level accuracy.

5. **Uncertainty-aware and probabilistic prediction.** Implementing Monte Carlo dropout or ensemble-based intervals  would enable clinically meaningful hypoglycaemia-alert thresholds with calibrated probabilities, complementing our point-accuracy benchmark.

6. **Broader method-transfer audits.** Applying the proposed multi-seed robustness-benchmarking protocol to other medical time-series tasks (ICU vitals, ECG, respiration) would test the generality of the "diminishing returns" phenomenon and build a shared evidence base for model-selection practice in medical AI [17, 20].

### 4.7 Conclusions

On the largest publicly available consolidated CGM cohort (1,092 people with T1D), a controlled multi-seed robustness-benchmarking protocol applied to four recurrent architectures shows that nominal architectural complexity — from a plain LSTM to a BiLSTM with attention — yields no clinically meaningful improvement in 30-minute, CGM-only glucose forecasting, with between-architecture RMSE differences (≤0.065 mg/dL) of the same order as, or smaller than, within-architecture seed variability. All deep models far outperform non-learning baselines, and all train cheaply. We interpret this as evidence that, for short-horizon univariate physiological forecasting, attainable accuracy is bounded less by model capacity than by the information content of the input channel. The practical implications are concrete for chronic-disease monitoring: for short-horizon glucose-management forecasting, a simple, reproducible LSTM is a sufficient and defensible baseline, and effort is better directed toward multimodal input, uncertainty quantification, external validation, and personalisation. Our controlled multi-seed robustness-benchmarking protocol offers a generalisable template for rigorous, reproducible comparisons in medical time-series AI.

---


---

## 5 Declarations

**Ethics approval and consent to participate.** This study is a secondary analysis of the publicly available MetaboNet dataset [15], which was collected within the framework of the original contributing clinical studies. No new human data were collected, and no additional ethical approval was required.

**Consent for publication.** Not applicable (secondary analysis of an anonymised, freely available public dataset).

**Availability of data and materials.** MetaboNet is publicly available at https://metabo-net.org. The custom data-loading pipeline, model implementations, evaluation scripts, and the clinical-evaluation scripts used in this study are released as a reproducible framework (including data loading, all four model architectures, the multi-seed training harness, statistical tests, and clinical evaluation) to support independent reproduction and reuse. The dataset is fully anonymised and aggregated from the original contributing clinical studies; no additional participant-level data are provided.

**Data availability.** The primary dataset (MetaboNet v2.0) used in this study is publicly accessible at https://metabo-net.org; the dataset paper is cited as reference [15]. This study performed a secondary analysis of this existing public resource; no new data were collected. The external T2D validation cohort was obtained from the publicly available ShanghaiT1DM/ShanghaiT2DM datasets [25], downloadable from figshare. **Code availability.** All analysis code — data preprocessing, the four model architectures, the 20-seed training harness, statistical tests, and clinical evaluation — is available in the project's public repository at https://github.com/nuohealth/metabonet-multiseed-audit to enable full reproduction of the results. **Access.** MetaboNet materials and the ShanghaiT2DM dataset are publicly downloadable without a data-use agreement.

**Reporting guideline.** This study is a methodological robustness-benchmarking comparison of recurrent architectures on a public dataset, rather than the development or validation of a clinical prediction model for individual-level use. Accordingly, the TRIPOD+AI statement (Collins et al. [24]) applies only partially; we follow its applicable items on data provenance, split integrity, and performance reporting where relevant to this non-deployment scope.

**Competing interests.** The authors declare no competing interests.

**Funding.** This work was supported by the corresponding author's institutional research programme. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

**Authors' contributions.** All authors contributed to study conception and design. ZS and PW designed the experimental protocol; ZS implemented the models and ran the experiments; ZS, YM, and XW performed statistical and clinical evaluation; all authors interpreted the results; ZS and PW drafted the manuscript; all authors critically revised the manuscript and approved the final version.

**Use of AI tools.** AI-assisted tools were used for parts of the manuscript drafting and language editing; the authors take full responsibility for the content.

---

## 6 Figure Legends

- **Figure 1.** Architectures are indistinguishable: seed-to-seed spread outweighs between-architecture differences. For each of the four recurrent architectures, the twenty independent random seeds (n = 20) are shown as points for the T1D MetaboNet cohort (blue, n = 1,092) and the T2D Shanghai cohort (red, n = 100), with the architecture mean marked by a bold tick and a dashed line tracing the mean trend. In each panel the vertical spread of the seed points (between-seed dispersion, T1D SD 0.05–0.07 mg/dL; T2D SD 0.13–0.44 mg/dL) exceeds the between-architecture gap in mean RMSE (0.065 mg/dL on T1D; 0.278 mg/dL on T2D), i.e., changing the random seed moves a given architecture more than changing the architecture does. All architectures degrade by a consistent +6.2–6.5 mg/dL when transferred to the harder T2D cohort. Within each cohort the architecture means are confined to a narrow band (16.23–16.30 mg/dL on T1D; 22.49–22.77 mg/dL on T2D).
- **Figure 2.** Model distinguishability as a forest plot of the 30-minute ΔRMSE (point estimate with bootstrap 95% CI, 1,000 resamples), for T1D (a) and T2D (b); positive favours the right-listed model. Red points denote contrasts significant after Holm–Bonferroni correction of the seed-level Diebold–Mariano test (T1D: LSTM vs. BiLSTM-Attention, Holm p = 0.014; LSTM-Attention vs. BiLSTM-Attention, Holm p = 0.022; T2D: LSTM vs. BiLSTM, p < 0.001; BiLSTM vs. LSTM-Attention, p = 0.014; LSTM vs. BiLSTM-Attention, p = 0.042). All point estimates fall well within ±1 seed-SD (≈0.07 mg/dL on T1D) and are ≈18–111-fold below the a priori clinically meaningful threshold (5 mg/dL) — statistically detectable yet clinically negligible.
- **Figure 3.** Added architectural complexity buys no meaningful accuracy. Relative (%) RMSE improvement over the plain-LSTM baseline of each architecture across all twenty seeds for T1D (blue) and T2D (red); shaded bands span the full 20-seed range. Added complexity yields at most ≈0.4% improvement on T1D and slightly degrades T2D (−0.2 to −1.2% by mean), confirming diminishing returns on CGM-only, 30-minute glucose forecasting.
- **Figure 4.** Computational efficiency: slower (more complex) models buy no accuracy, with RMSE pinned near a single cohort level. Training time vs. accuracy for all 160 training runs (4 architectures × 20 seeds × 2 cohorts) on independent time axes per cohort; dotted line marks the cohort mean RMSE. T2D (100 subjects) trains about ten times faster (mean ~7.5 min; range 4.8–10.9 min) than T1D (1,092 subjects; 42–186 min) with the same four architectures, and all models sit on the same RMSE level regardless of training cost.
- **Figure 5.** Clinical equivalence: all architectures pass consensus thresholds and are clinically indistinguishable. (a) Mean absolute relative difference (MARD, 20-seed mean ± SD) against the consensus <10% acceptability threshold (green band); error bars overlap substantially, so the four architectures are clinically indistinguishable on MARD (7.33–7.43%). (b) Clarke error-grid Zone A (%) against a narrow reference band; all four architectures cluster within 77.8–79.3% Zone A. Because the architectures are clinically equivalent and all pass consensus, the simplest LSTM is a sufficient choice.
---

## 7 Tables

**Table 1.** Model performance comparison on the MetaboNet test set at the 30-minute prediction horizon (20-seed mean ± SD; 219 independent test subjects / 219,000 windows). Train time is the mean over 20 seeds. Baselines: persistence RMSE 27.37 mg/dL; linear extrapolation 26.64 mg/dL.

**Table 2.** Pairwise statistical comparison of RMSE across architectures (n = 20 seeds). ΔRMSE = RMSE(A) − RMSE(B); positive favours B (right-listed model). Effect size Cohen's d (pairwise dz); p-values from paired t-test with Holm–Bonferroni correction (6 pairs); bootstrap 95% CI of the RMSE difference (1,000 resamples).

---

## 8 References

[1] International Diabetes Federation. IDF Diabetes Atlas. 10th ed. Brussels: International Diabetes Federation; 2021.

[2] Rodbard D. Continuous glucose monitoring: a review of recent studies demonstrating improved glycemic outcomes. Diabetes Technol Ther. 2017;19(S3):S25–S37.

[3] Bergenstal RM, Klonoff DC, Garg SK, et al. Threshold-based insulin-pump interruption for reduction of hypoglycemia. N Engl J Med. 2013;369(3):224–232.

[4] Battelino T, Danne T, Bergenstal RM, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. Diabetes Care. 2019;42(8):1593–1603.

[5] Pérez-Gandía C, Facchinetti A, Sparacino G, Cobelli C, Gómez EJ. Artificial neural network algorithm for online glucose prediction from continuous glucose monitoring. Diabetes Technol Ther. 2010;12(1):81–88.

[6] Zhu T, Li K, Herrero P, Georgiou P. Deep learning for diabetes: a systematic review. IEEE J Biomed Health Inform. 2021;25(7):2744–2757.

[7] Li K, Daniels J, Liu C, Herrero P, Georgiou P. Convolutional recurrent neural networks for glucose prediction. IEEE J Biomed Health Inform. 2020;24(2):568–577.

[8] Zhu T, Li K, Herrero P, Georgiou P. Personalized blood glucose prediction for type 1 diabetes using evidential deep learning. IEEE Trans Biomed Eng. 2023;70(1):283–294.

[9] Mirshekarian S, Bunescu R, Marling C, Schwartz F. Using LSTMs to learn physiological models of blood glucose behavior. In: Proc Annu Int Conf IEEE Eng Med Biol Soc (EMBC). 2017:2887–2891.

[10] Mirshekarian S, Shen H, Bunescu R, Marling C. LSTMs and neural attention models for blood glucose prediction: comparative experiments on real and synthetic data. In: Proc Annu Int Conf IEEE Eng Med Biol Soc (EMBC). 2019:706–712.

[11] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate. arXiv:1409.0473. 2015.

[12] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Adv Neural Inf Process Syst. 2017;30:5998–6008.

[13] Sarwar MA, Maqsood S, Belousoviene E, et al. AutoBiGluNet: transformer-based time series modeling for blood glucose prediction. Health Inf Sci Syst. 2026. doi:10.1007/s13755-026-00469-4.

[14] Liu K, Li L, Ma Y, Jiang J. Machine learning models for blood glucose level prediction in patients with diabetes: a systematic review. JMIR Med Inform. 2023;11:e47833.

[15] Wolff MK, Calhoun P, Aiello EM, Qin Y, Royston SF. MetaboNet: the largest publicly available consolidated data set for type 1 diabetes management. J Diabetes Sci Technol. 2026. doi:10.1177/19322968261441637.

[16] Liu C, Vehí J, Avari P, Reddy M, et al. Long-term glucose forecasting using a physiological model and deconvolution of the continuous glucose monitoring signal. Sensors. 2019;19(19):4338.

[17] Yen HK, Yang JJ, Groot OQ, et al. Fostering reproducibility and generalizability in machine learning for clinical prediction modeling. Spine J. 2023;23(2):191–200.

[18] Ribeiro AH, Ribeiro MH, Paixão GMM, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat Commun. 2020;11:1760.

[19] Zhang Q, Zhou H, Zhu X, Lin R, Hu L, Zhu G. Transforming hypoglycemia prediction in adult type 1 diabetes: a systematic review and meta-analysis for precision care. Open Life Sci. 2026;21(1):20251325. doi:10.1515/biol-2025-1325.

[20] Makridakis S, Spiliotis E, Assimakopoulos V. Statistical and machine learning forecasting methods: concerns and ways forward. PLoS One. 2018;13(3):e0194889.

[21] Diebold FX, Mariano RS. Comparing predictive accuracy. J Bus Econ Stat. 1995;13(3):253–263.

[22] Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating clinical accuracy of systems for self-monitoring of blood glucose. Diabetes Care. 1987;10(5):622–628.

[23] Marling C, Bunescu R. The OhioT1DM dataset for blood glucose level prediction: update 2020. CEUR Workshop Proc. 2020;2675:71–74.

[24] Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.

[25] Zhao Q, et al. Chinese diabetes datasets for data-driven machine learning. Sci Data. 2023;10(1):35. doi:10.1038/s41597-023-01940-7. [ShanghaiT1DM/ShanghaiT2DM datasets]

[26] Jeffries N, Wolff M, Royston S, et al. MetaboNet-Bench: A multi-modal benchmark for glucose forecasting in type 1 diabetes. ArXiv:2606.18640 [preprint]. 2026 Jun 25.

[27] Wolff MK, Schaathun HG, Gros S, Volden R, Steinert M, Fougner AL. Blood glucose prediction algorithms require clinically relevant performance criteria beyond accuracy. Diabetes Technol Ther. 2025;27(10):858–870. doi:10.1089/dia.2025.0074.

[28] International Organization for Standardization. In vitro diagnostic test systems — Requirements for blood-glucose monitoring systems for self-testing in managing diabetes mellitus. ISO 15197:2013. Geneva: ISO; 2013.

