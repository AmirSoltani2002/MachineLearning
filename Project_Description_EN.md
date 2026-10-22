# Machine Learning Final Project

University of Tehran - College of Engineering - School of Electrical and Computer Engineering

**Instructors:** Dr. Aarabi and Dr. Abolghasemi

**Original document date:** Azar 1403 (Iranian calendar)

**Document type:** English translation of the original Persian assignment brief

This translation preserves the assignment requirements and historical deadlines. It describes what students were asked to do, rather than asserting what this repository implements. The original page layout and table-of-contents page numbers have not been reproduced. English technical terms, punctuation, and an apparent MFCC typographical error have been normalized for readability.

## Contents

1. Introduction
2. Problem definition and project description
   - 2.1 General problem definition
   - 2.2 Dataset
   - 2.3 Team formation
   - 2.4 Preliminary report
   - 2.5 Data preprocessing and feature extraction
   - 2.6 Classification
   - 2.7 Clustering
3. Project report
4. Grading
5. Final notes

## 1. Introduction

Machine learning is a key tool for analyzing and understanding patterns in data. It is used in many areas, including recognition and authentication. This project introduces students to fundamental machine-learning concepts through practical work on two audio-related tasks: speaker gender classification and authentication using voice characteristics.

You will design a machine-learning model that first predicts a speaker's gender from time-domain or frequency-domain audio features, and then identifies the speaker using these features. Speech recordings contain a variety of characteristics that differ between individuals. To complete the project successfully, you must preprocess the data to remove noise, extract useful audio features using machine-learning and audio-signal-processing methods, and classify and cluster the data using those features. In addition to introducing machine-learning principles, this process develops practical experience with challenges such as selecting appropriate features and improving model accuracy.

## 2. Problem definition and project description

### 2.1 General problem definition

Speech-signal analysis and processing involve studying audio signals and extracting useful information from them. These methods have many applications. After initial audio preprocessing, different features can be extracted and used by machine-learning algorithms for tasks such as speaker recognition, emotion recognition, understanding spoken sentences, and speech-to-text conversion. Speech signals contain valuable information in both the time and frequency domains, allowing different processing tools to be developed. This project focuses on speaker gender classification and speaker identification.

The project has two phases:

**Phase 1: Background study.** Become familiar with audio signals, particularly speech signals, and their processing applications. Prepare a short report describing speech-signal data and the challenges of recognizing patterns in these signals. For each challenge, briefly explain relevant solutions and discuss their advantages and disadvantages, citing reliable sources. For example, Mel-frequency cepstral coefficients (MFCCs) are an important method of extracting features from speech. Explain such methods, the challenges they address, and the strengths and weaknesses of each approach. Further details follow below.

**Phase 2: Practical implementation.** Implement the stages described below using the supplied dataset. The data consists of speech recordings collected during the semester as students explained their homework solutions. First preprocess the recordings into an appropriate format and remove noise or unwanted interference that could affect later stages. Extract features using the methods introduced below or other methods investigated in Phase 1. Finally, classify and cluster the data. If the extracted features do not support satisfactory results, investigate better features to improve accuracy.

### 2.2 Dataset

The supplied folder contains audio recordings of different lengths, using the following filename format:

```text
HWx_Qy_Student-ID_gender.mp3
```

For each task, including gender classification and speaker identification, organize the dataset into appropriate categories using gender labels or student IDs, then prepare and preprocess the recordings for feature extraction.

Because the recordings have different lengths, divide them into shorter time windows using a suitable method and use some of these windows in later stages. No particular window length or segmentation method is mandatory. You may convert MP3 recordings to another format if useful. Check each recording's sampling rate and ensure that all recordings are processed at the same rate.

You do not have to use the entire dataset because of its size. The requirements for each part are explained below. Additional homework recordings will be provided gradually.

**Original course dataset link:**

https://drive.google.com/drive/folders/1pq_jGqdBda_QjNnK2yAzD4N2grbPF8Rs?usp=sharing

This link is reproduced from the assignment; its present availability has not been verified.

### 2.3 Team formation

You may work individually or in a team of up to four members. One representative must submit the team members' names in the designated team-registration area by the end of **20 Azar**.

### 2.4 Preliminary report

Prepare a preliminary report as a team. Its purpose is to introduce voice authentication, its challenges, and methods for preprocessing audio and extracting features. Explain closed-set and open-set authentication, including their implementation challenges. The report should build an understanding of developments and techniques in this area and discuss gender classification as a relevant application.

Include the following topics:

#### 1. Introduction to voice authentication

- Define voice authentication and explain its importance.
- Review applications such as speaker identification and speaker gender classification.
- Define closed-set and open-set authentication and explain their differences.
- Discuss how these two approaches can be implemented.
- Discuss their applications in voice authentication.

#### 2. Challenges in voice authentication

- Identify and explain the main challenges in research and practical applications of voice authentication and gender classification.
- Review potential solutions and ongoing research addressing these challenges.

#### 3. Audio preprocessing

- Discuss why preprocessing matters for voice authentication and gender classification.
- Explain common preprocessing stages, focusing on noise reduction, normalization, and windowing.

#### 4. Feature-extraction techniques

Explain techniques used in audio analysis, including:

- MFCCs: Mel-frequency cepstral coefficients.
- Fast Fourier transform (FFT).
- Log-Mel spectrograms.
- Spectral centroid.
- Chroma features.
- Spectral contrast.
- Zero-crossing rate.
- Linear predictive coding (LPC).
- Perceptual linear prediction (PLP).

#### 5. Similarity learning - bonus topic

- Define similarity learning in audio analysis and explain how it can measure similarity between audio features.
- Introduce and explain common loss functions, such as contrastive loss and triplet loss.

### 2.5 Data preprocessing and feature extraction

Learn how to prepare audio data and apply feature-extraction methods to understand speech-signal characteristics. Before starting, exploratory analysis, such as inspecting frequency spectra, can help select appropriate preprocessing and feature-extraction techniques.

#### 1. Load the data

- Load the raw audio recordings.

#### 2. Preprocess the audio

- If noise is present, consider techniques such as spectral subtraction or bandpass filtering. Whether these techniques are useful depends on the data and project needs; choose a method based on analysis of the noise and signal characteristics.
- Ensure that all recordings are processed at the same sampling rate. Resample when necessary.
- Divide signals into shorter, analyzable time windows using tools such as `librosa` or `scipy.signal`.
- Normalize the signals so that recordings are consistent and comparable.

#### 3. Extract audio features

- Extract frequency-related features such as MFCCs, log-Mel spectrograms, spectral centroid, spectral bandwidth, and spectral contrast.
- Use time-domain features such as zero-crossing rate and energy to examine rapid changes, temporal variation, and signal intensity.
- Combine time-domain and frequency-domain features if needed to enrich the feature set.

#### 4. Visualize the audio data

- Use spectrograms or Mel spectrograms to visualize spectral information.
- Use dimensionality-reduction methods such as t-SNE or PCA to visualize the data in two or three dimensions and analyze classification or clustering structure.

### 2.6 Classification

Use the features extracted in the previous stage for classification. Complete both speaker gender classification and speaker identification using closed-set authentication. The feature sets may differ between the two tasks.

**Gender classification:** Use recordings from at least **25% of the individuals** in the dataset. You may use only selected segments from their recordings. Keep the two classes balanced.

**Closed-set speaker identification:** Randomly select recordings from at least **six people**, with balanced gender representation, and train the models on these speakers. Repeat this process **three times**, using different randomly selected groups of at least six people. Include the results and analysis for all three groups in the report.

For each task:

1. Divide the data into training and test sets. The test set must contain at least **25%** of the data. Preserve class proportions in both sets.
2. Apply at least **three different machine-learning methods** covered during the course. Compare and analyze their results. Code alone is insufficient: analysis of data plots and comparison of the methods are particularly important. Possible methods include logistic regression, multilayer perceptrons (MLPs), k-nearest neighbors (KNN), support vector machines (SVMs), AdaBoost, XGBoost, or another appropriate method.
3. You may apply normalization when using these methods.
4. For each classifier, include a confusion matrix, ROC curves, an analysis of errors for each class, and any other necessary evaluation material.

### 2.7 Clustering

Cluster the data using at least **two methods** covered in the course. Use a criterion such as the silhouette score to find a suitable number of clusters and plot the results. Then perform clustering using that selected number and **two additional cluster-count settings**, and analyze the outcomes.

Carefully examine similarities within clusters, differences between clusters, and why particular samples are assigned to the same cluster. Analysis and reporting are especially important in this section.

## 3. Project report

Correct code must be accompanied by a detailed report with clear explanations and careful analysis of the data and plots. Include all relevant observations from the project.

The preliminary report is due by **30 Azar** and serves as the introduction to the final report. Submit the final report, code, and required analyses by **25 Dey**.

The final report must:

- Fully explain the preprocessing methods used.
- Include the required clustering analyses for different numbers of clusters.
- Explain why the classification models were selected.
- Report evaluation metrics such as precision, recall, and F1 score for every trained model.
- Include all important code sections and explain them. The report is the final basis for grading; if a section's code is not explained in the report, half of that section's marks will not be awarded.
- Be well organized, divided into sections, and supported by detailed analysis.
- Include code as properly formatted text rather than screenshots.

Both the preliminary and final reports may be written in **Persian or English**.

## 4. Grading

| Component | Marks out of 100 |
|---|---:|
| Preliminary report | 20 |
| Preprocessing and feature extraction | 30 |
| Classification | 30 |
| Clustering | 20 |
| Total | 100 |

## 5. Final notes

- Similarity between the work of different individuals or teams is not accepted. Any detected cheating will be handled according to the course rules.
- You may use sources with appropriate citations. However, a report that directly translates a source, or uses another person's report, is considered cheating.
- Read this brief fully and carefully before emailing the project designers with questions.

**Teaching-assistant contacts, as listed in the original assignment:**

- Kiana Hooshanfar: k.hooshanfar@ut.ac.ir
- Reza Rezaeian: r.rezaeian@ut.ac.ir
- Mohammad Valinezhad: m.valinezhad@ut.ac.ir
