<h1 align=center> Exploring Data Augmentation Methods through Attribution

![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/transformers-4.40.1-blue) ![](https://img.shields.io/badge/captum-0.7.0-blue) ![](https://img.shields.io/badge/torch-2.3.0-blue) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

The following research explores data attribution methods to identify training subsets from a larger training set with varying contributions to a model's performance. Two approaches have been tested here - (1) Captum's Layer Integrated Gradients for feature attribution in text classification, and (2) a self-implemented perplexity-based text quality filtering method inspired by a recent Meta research paper for translation tasks. Additionally, we examine the effects of applying data augmentation to the top 10% and bottom 10% of training samples in both tasks using established techniques, including Easy Data Augmentation (EDA), An Easier Data Augmentation (AEDA), backtranslation (English ↔ French), and paraphrasing with the instruction-following variant of the Llama 3.1 (8B) model.

## Project Structure

```html
|- assets
|- attribution
  |-- lig.py
  |-- perplexity.py
|- augmentation
  |-- An Easier Data Augmentation
  |-- Backtranslation
  |-- Easy Data Augmentation
  |-- Paraphrasing
|- datasets
|- plots
|- train
  |-- bert.py
  |-- mT5.py
```

## Datasets

The datasets for both tasks are available as .csv files in this repository.

- **Task 1 - Classification**

  - SST2 - binary classification dataset
  - SST5 - multi-label classification dataset
  - IMDB - binary classification dataset
  - AGNews10K - multi-label classification dataset

- **Task 2 - Language Translation**

  - English to Afrikaans
  - English to Welsh
  - English to Czech
  - English to Spanish
  - English to Romanian

## Methodology

- **Task 1 - Classification**

<p align="center">
  <img src = "assets/readme/classification methodology.png" max-width = 100% height = '400' />
</p>

- **Task 2 - Language Translation**

<p align="center">
  <img src = "assets/readme/translation methodology.png" max-width = 100% height = '403' />
</p>