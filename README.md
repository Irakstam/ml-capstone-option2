# Does a neural network beat logistic regression on Adult income data?

This project compares two approaches to predicting whether a person's annual income exceeds $50,000 using the Adult Census Income data set. The objective is to evaluate whether a small neural network improves on logistic regression enough to justify its additional architectural and computational complexity. The hypothetical setting is a public agency considering automated targeting for household benefits, where the two kinds of classification error have different consequences.

The [completed notebook](Final_Project_Option2_Tabular_NN_vs_Classical.ipynb) contains the executed code, evaluation plots, and detailed interpretations. A standalone [HTML report](results/option2_report.html) with the same pre-rendered outputs is also in the repository.

The data originate from the [UCI Adult data set](https://archive.ics.uci.edu/dataset/2/adult), accessed via [OpenML Adult version 2 (ID 1590)](https://www.openml.org/d/1590). The dataset contains 48,842 individuals and 14 input features. The positive class (`>50K`) makes up 23.9% of the records, so the classes are substantially imbalanced. Missing values are present in `workclass`, `occupation`, and `native-country`.

## Methodology

A stratified 80/20 train/test split was used (39,073 training rows, 9,769 test rows). Numerical features were imputed with the training median and standardised; categorical features were imputed with the most frequent training category and one-hot encoded.

Logistic regression sits inside a single scikit-learn `Pipeline`, so its preprocessing is refitted independently within each fold of a 5-fold stratified cross-validation on the training set, and again when the pipeline is fitted on the full training set for the test evaluation.

The neural network was built with the Keras Sequential API and has two dense hidden layers (64 and 32 ReLU units), dropout (0.3), and a sigmoid output unit. Because Keras takes the last 10% of the rows it is given for validation, the network's preprocessing is fitted only on the first 90% of the training rows; that keeps the validation rows out of the imputation, scaling, and category vocabulary. Training used Adam and binary cross-entropy, with early stopping restoring the weights from the lowest validation-loss epoch (epoch 2).

## Key results

Both models were evaluated on the identical held-out test set (9,769 samples). Because the higher-income bracket is the minority class, F1 score is the primary benchmark:

| Model | Test F1 | Test accuracy | Precision | Recall | False positives | False negatives |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression** | 0.6562 | 0.8524 | 0.7414 | 0.5885 | 480 | 962 |
| **Neural network** | 0.6809 | 0.8590 | 0.7430 | 0.6283 | 508 | 869 |

The neural network improved test F1 by **+0.0247** and test accuracy by **+0.67 percentage points**. The gain appears mainly in recall rather than precision: 93 additional individuals in the higher-income bracket were identified (false negatives fell from 962 to 869), at the cost of 28 additional false positives (480 to 508). The [logistic regression](results/figures/logistic_regression_confusion_matrix.png) and [neural network](results/figures/neural_network_confusion_matrix.png) confusion matrices show this trade-off, and the [learning curves](results/figures/training_curves.png) show validation loss reaching its optimum at epoch 2, with early stopping halting training at epoch 5. Five-fold cross-validation F1 for logistic regression averaged 0.6576 with a fold standard deviation of 0.0098, so the baseline is stable across folds.

## Critical interpretation and ethical considerations

Accuracy alone is a misleading metric for this dataset: a trivial majority-class classifier predicting `<=50K` for every record achieves 76.07% test accuracy, yet yields an F1 score of 0.0000 for the positive class.

On tabular data with one-hot encoded categories, linear models provide strong baselines. While the neural network can represent non-linear feature interactions, the modest +0.0247 F1 improvement comes at an 84-fold parameter increase (8,897 weights vs. 106).

The dataset records historical individual attributes, including sex, race, and native country, so models trained on it risk perpetuating systemic disparities. In a public agency benefit allocation context, the cost asymmetry between false positives (potentially misallocating resources or denying needs-based aid) and false negatives (failing to reach eligible households) must guide model selection and threshold determination. An interpretable classical model may be preferable for transparency and procedural fairness unless extensive demographic parity audits justify the complex model.

These results come from a single train/test split and a single network seed, so the size of the F1 gain should not be read as a fixed property of the two model families.

## Environment setup

To run or inspect the notebook locally with Python 3.11:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter to view or rerun the notebook
jupyter notebook Final_Project_Option2_Tabular_NN_vs_Classical.ipynb
```

## Repository structure

```
├── Final_Project_Option2_Tabular_NN_vs_Classical.ipynb  # Executed capstone notebook (all 38 cells run with outputs visible)
├── README.md                                            # Project summary, methodology, results, and critical reflection
├── requirements.txt                                     # Python dependencies
└── results/
    ├── option2_report.html                              # Standalone pre-rendered HTML export for offline reading
    ├── model_comparison.csv                             # Exported comparative performance metrics
    ├── metrics.json                                     # Full serialized experiment metrics and configuration
    └── figures/
        ├── logistic_regression_confusion_matrix.png     # Test confusion matrix (classical baseline)
        ├── neural_network_confusion_matrix.png          # Test confusion matrix (neural network)
        ├── training_curves.png                          # Training vs validation loss and early-stopping checkpoint
        └── model_comparison.png                         # F1 and accuracy comparison bar chart
```
