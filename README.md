# Gender Classification and Speaker Identification from Audio

A University of Tehran machine-learning course project by **Amirabbas Reza Soltani** and **Arad Vazirpanah**. The project explores how handcrafted speech features support binary gender-label classification, identification among known speakers, and unsupervised clustering.

The notebook combines audio preprocessing, MFCC and log-Mel feature extraction, PCA, classical classifiers, and small PyTorch neural networks. The accompanying report explains the methods, compares the saved experiments, and examines why audio clusters do not necessarily correspond to individual speakers.

**Start here:** [Project report](Report/Report.pdf) | [Notebook](Code/ML_FinalProject.ipynb) | [Assignment brief in English](Project_Description_EN.md) | [Printable brief](Project_Description_EN.pdf)

## Project scope

| Task | Approach |
|---|---|
| Gender-label classification | Predict the dataset's `female`/`male` labels using logistic regression, KNN, SVM, and an MLP. |
| Closed-set speaker identification | Classify samples among ten selected known speakers; repeat with three speaker groups. |
| Clustering | Compare K-means with a custom sequential, centroid-distance clustering algorithm using silhouette scores. |

The assignment and report call the identification task **closed-set authentication**. Here, this means choosing among known speakers; it does not implement rejection of unknown speakers. Open-set authentication and similarity-learning losses are discussed in the report's background, rather than implemented as a deployed authentication system.

## Repository contents

```text
.
├── Code/
│   └── ML_FinalProject.ipynb       # Original notebook and saved outputs
├── Data/
│   └── features_pca.csv           # Cached PCA features and labels
├── Report/
│   └── Report.pdf                 # Original English project report
├── Project_Description_EN.md      # Translated assignment requirements
├── Project_Description_EN.pdf     # Printable English translation
├── README.md
└── .gitignore
```

The English assignment translation preserves the original requirements and historical course deadlines. It is separate from the implementation summary in this README.

## Data and processing pipeline

The original input consists of students' spoken homework explanations, with gender labels and student IDs encoded in filenames. The raw recordings are **not included** in this folder. The supplied CSV contains **6,147 rows**, **219 PCA features** (`PC1` through `PC219`), and the label columns `ID` and `Gender`. It represents **113 distinct speaker IDs**, with 4,573 rows labeled `1` (male) and 1,574 labeled `0` (female).

The notebook's audio-to-feature workflow is:

1. **Read filenames and labels.** Extract the speaker ID and gender label from MP3 filenames.
2. **Explore and denoise recordings.** Inspect waveforms, spectrograms, and a heuristic noise estimate; apply a fifth-order Butterworth bandpass filter from 100 to 8,000 Hz at the original sampling rate, followed by spectral subtraction.
3. **Resample and segment.** Resample to 1,000 Hz, split recordings into complete ten-second segments, and peak-normalize each segment. Incomplete trailing segments are discarded.
4. **Extract features.** Stack 13 MFCC channels and 128 log-Mel channels over time, flatten them, and append signal energy. The report records 2,821 features per segment.
5. **Standardize and reduce dimensions.** Use PCA for three-dimensional visualization and retain 219 components to explain 95% of the observed variance.
6. **Classify and cluster.** Train supervised baselines and explore clustering structure using the resulting feature vectors.

The report's sections 3 and 4 describe preprocessing and feature extraction. The 1,000 Hz resampling choice is part of the original experiment, not a recommended full-band speech-processing setting: it limits retained frequencies to approximately 500 Hz.

## Experiments and saved results

The following values come from the existing report and notebook outputs; training was not rerun while preparing this documentation. **The evaluation limitations below affect their interpretation, especially the speaker-identification results.**

### Gender-label classification

The experiment balances the selected data to 1,300 samples per label and uses a stratified 75%/25% train/test split, giving a test set of 650 samples. A further `StandardScaler` is fitted on the training partition. The PyTorch MLP has hidden layers of 64 and 32 units, ReLU activations, and a sigmoid output; it is trained with binary cross-entropy and Adam for 100 epochs.

| Model | Correct predictions / 650 | Accuracy from saved confusion matrix |
|---|---:|---:|
| Logistic regression | 644 | 99.08% |
| K-nearest neighbors | 560 | 86.15% |
| Support vector machine | 642 | 98.77% |
| Multilayer perceptron | 629 | 96.77% |

Section 5.1 of the report includes per-class precision, recall, F1 scores, confusion matrices, and ROC curves. These are within-dataset sample-split results, not evidence of generalization to unseen speakers or recording conditions.

### Closed-set speaker identification

For each of three trials, the notebook selects five male and five female speakers with more than 50 samples each, equalizes sample counts across speakers, and applies a stratified 25% test split. The report records the following rounded accuracies:

| Model | Trial 1 | Trial 2 | Trial 3 |
|---|---:|---:|---:|
| Logistic regression | 100% | 100% | 100% |
| K-nearest neighbors | 64% | 75% | 83% |
| Support vector machine | 96% | 99% | 99% |
| Multilayer perceptron | 93% | 96% | 95% |

**These are historical outputs with target leakage, not valid measures of voice-based identification.** In `get_data`, speaker IDs are one-hot encoded, then `X` is formed by dropping only `Gender`. Consequently, the one-hot ID columns used to derive the target are also included in the input. Those columns must be excluded and the experiments rerun before making identification-performance claims.

### Clustering

The notebook searches sequential-clustering distance thresholds from 30 to 145 in steps of five, and K-means cluster counts from 2 to 112 in steps of five. The report identifies a sequential threshold of 145 and K-means `k=2` as the best silhouette settings among those evaluated.

- K-means produces two clusters that are associated with the binary gender labels: approximately 94.3% of female-labeled samples fall in one cluster, and 86.5% of male-labeled samples in the other.
- The sequential algorithm's two-cluster result does not show the same label separation. The report considers outliers, recording conditions, and sample-order sensitivity as possible explanations.
- Experiments with 113 K-means clusters and a sequential threshold near 70 do not produce a clean one-cluster-per-speaker partition.

These are exploratory findings from report section 6, not a validated speaker-clustering benchmark.

## Running or reviewing the project

### Environment

The notebook metadata records Python **3.8.6**. Dependency versions were not pinned in the original submission, and a fresh execution environment has not been validated. Its main libraries are NumPy, pandas, SciPy, librosa, SoundFile, Matplotlib, scikit-learn, and PyTorch.

A starting environment can be prepared from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy librosa soundfile matplotlib scikit-learn torch jupyterlab
cd Code
jupyter lab ML_FinalProject.ipynb
```

Starting Jupyter from `Code/` preserves the notebook's `../Data/...` relative paths. The package list is a setup starting point, not a tested lockfile.

### Review the cached-feature experiments

The CSV allows inspection of the classification experiments without the missing audio. In the notebook, start at the **Classification** section: execute its imports, NumPy seed, and `pd.read_csv("../Data/features_pca.csv")` cell, then continue through the gender-classification cells. The later **closed-set authentication** section preserves the original experiment and its leakage issue described above.

### Rebuild features from audio

This route needs the original course recordings in `Data/HW1_M/`. The notebook writes intermediate audio into `Data/Denoized/`, `Data/resampled/`, and `Data/Samples/`. The course dataset link is reproduced in the [translated assignment](Project_Description_EN.md); access and present availability have not been verified.

**The notebook is an archived submission and is not currently runnable end to end without corrections.** In particular:

- The feature-extraction loop is missing a colon after `for filename in os.listdir(dir)` and refers to a Google Colab path that must be replaced with a local path.
- `PCA`, `KMeans`, `silhouette_score`, and `matplotlib.patches as mpatches` are used without corresponding imports in the saved notebook.
- `process_and_save_files` calls `os.makedirs(output_dir)` without `exist_ok=True`, so rerunning that step requires an output-directory check.
- Segment filenames omit the source-recording index. Different recordings from the same speaker can therefore overwrite each other's numbered segments.

No code changes or new model results are implied by this documentation.

## Evaluation limitations and next steps

Beyond the speaker-ID target leakage, several issues should be addressed before drawing new performance conclusions:

- **Preprocessing leakage:** the original feature standardization and PCA are fitted before the supervised train/test split. Fit them on training data only, then transform validation/test data.
- **Related samples across partitions:** the split operates on segments, without grouping by original recording. For gender classification, use speaker-disjoint evaluation; for known-speaker identification, keep speakers represented while separating recording sessions or source files across partitions.
- **Speaker selection:** the gender-classification code samples from row-level ID arrays, not unique speaker lists. Its stated selection of 50 distinct speakers per gender is therefore not guaranteed.
- **Audio preprocessing:** verify the low sampling rate, ensure filter cutoffs are below Nyquist, and revisit the spectral-subtraction noise window, whose current indexing uses samples as though they were STFT frames. The reported noise estimate is a heuristic rather than a calibrated SNR measurement.
- **Reproducibility:** NumPy and some split seeds are specified, but PyTorch is not seeded and dependencies are unpinned. Save source-recording identifiers, fitted transforms, environment versions, and corrected experiment settings for a reproducible rerun.

The most useful next step is a clean, leakage-free baseline using only audio-derived features, followed by re-evaluation under grouped splits. The original notebook and report are retained as the record of the submitted coursework.
