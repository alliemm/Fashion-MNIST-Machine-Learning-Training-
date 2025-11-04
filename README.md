# Fashion-MNIST Dataset Analysis and Classification

## Introduction
Our team applied machine learning techniques to a fashion-oriented classification task using the [Fashion-MNIST dataset](https://paperswithcode.com/dataset/fashion-mnist) from Papers With Code. This dataset contains 70,000 grayscale images of 28×28 pixels across 10 categories, with 7,000 images per category. The dataset is split into a training set of 60,000 images and a test set of 10,000 images.  

The goal of this project is to accurately classify these clothing images using various supervised learning models, including logistic regression, neural networks, and convolutional neural networks (CNNs).  

---

## Unsupervised Analysis
Before applying supervised learning, we conducted unsupervised analysis to explore the dataset's structure. Techniques such as Principal Component Analysis (PCA) were used, but no clear clusters or patterns were observed.  

This step helped in understanding data distribution and variance, and it also contributed to reducing epoch times for subsequent model training. Note that this unsupervised analysis was applied only to logistic regression and neural networks.  

---

## Supervised Analysis

### Logistic Regression
Logistic regression served as a baseline model. We experimented with L1 and L2 regularization and analyzed their effects on training and test performance.  

| Model                | Regularization | Training Accuracy | Test Accuracy | Precision | Recall |
|--------------------- |---------------|-----------------|--------------|-----------|--------|
| Logistic Regression  | L1            | 0.51            | 0.495        | 0.484     | 0.496  |
| Logistic Regression  | L2            | 0.52            | 0.51         | 0.49      | 0.51   |
| Logistic Regression  | None          | 0.4947          | 0.498        | 0.495     | 0.498  |

- L2 regularization provided the highest training and test accuracies, indicating effective generalization.  
- Precision and recall were relatively consistent across regularization techniques.  

---

### Neural Networks
Neural networks were designed with varying hidden layers and activation functions. We tuned the learning rate (~0.001) and analyzed overfitting and underfitting scenarios.  

| Model           | Hidden Layers | Activation Function | Training Accuracy | Test Accuracy | Precision | Recall |
|-----------------|---------------|-------------------|-----------------|---------------|-----------|--------|
| Neural Networks | 1             | ReLU              | 0.9475          | 0.875         | 0.875     | 0.875  |
| Neural Networks | 2             | ReLU              | 0.9391          | 0.874         | 0.8749    | 0.8743 |
| Neural Networks | 3             | ReLU              | 0.9326          | 0.8737        | 0.876     | 0.878  |

- A single hidden layer achieved the highest performance.  
- Precision and recall remained consistent across layer configurations, indicating robust classification.  

---

### Convolutional Neural Networks (CNN)
CNNs were tested with different architectures, activation functions, and kernel sizes. We optimized the learning rate (~0.0015) and analyzed overfitting and underfitting.  

| Model | Architecture  | Activation Function | Training Accuracy | Test Accuracy | Precision | Recall |
|-------|---------------|-------------------|-----------------|---------------|-----------|--------|
| CNN   | Architecture A | ReLU              | 0.9387          | 0.895         | 0.895     | 0.895  |
| CNN   | Architecture B | Sigmoid           | 0.8765          | 0.8744        | 0.8744    | 0.8744 |

- Architecture A with ReLU outperformed Architecture B in accuracy, precision, and recall.  

---

## Analytical Discussion
- CNNs achieved the best accuracy and generalization, highlighting the advantage of convolutional operations in capturing spatial patterns in images.  
- Logistic regression struggled with complex, non-linear patterns, while traditional neural networks performed better but could not capture spatial information as effectively as CNNs.  
- Overall ranking of model effectiveness: **CNN > Neural Networks > Logistic Regression**.  
- However, maintaining spatial information was less critical in Fashion-MNIST because all images are centered and subjects fill the frame.  

---

## Conclusion
While CNNs achieved the highest accuracy, neural networks are recommended for this dataset due to their efficiency and lower computational cost.  

Future improvements could include:
- Advanced CNN architectures (e.g., residual networks, attention mechanisms)  
- Data augmentation  
- Transfer learning  

These techniques can enhance model generalization and robustness across all models.  

---
