# Unsupervised Learning with Competing Hidden Units – Enhanced!

Welcome to my repository! This project extends the insights from the research paper *Unsupervised Learning with Competing Hidden Units*, taking their biologically-inspired, unsupervised learning concepts and pushing them to the next level. If you're as excited about biologically-plausible learning as I am, you're in the right place!

---

### 🧠 Quick Summary of the Paper
The original paper introduces a novel approach to unsupervised learning, where competition among hidden units in a neural network helps the model learn meaningful patterns from data without labels. Instead of relying on traditional error-backpropagation, this biologically-motivated method uses local learning rules to enable self-organization of hidden units, which ultimately helps in capturing complex features in data.

---

### 🚀 What’s New in This Repository? Extensions to the Original Work
In this repository, I extend the paper's work in three exciting directions:

1. **Diverse Datasets:** I train and test models on **MNIST**, **CIFAR-10**, and **FASHION-MIST** datasets, covering a broader range of data patterns and complexities.
   
2. **Multi-Layer Unsupervised Learning:** I explore the effectiveness of unsupervised learning across multiple hidden layers, followed by a final supervised layer, to see if deeper unsupervised architectures enhance learning.

3. **Synaptic Specialization:** I dive into biologically-plausible dynamics by experimenting with networks that have **only excitatory or only inhibitory synapses** in hidden layers, adding another layer of bio-inspired realism.

---

### 📂 Repository Contents

Here's an overview of the main components in this repository:

1. **`backpropagation.ipynb`**  
   * A Google Colab-ready notebook to train a simple feedforward neural network on **MNIST**, **CIFAR-10**, and **SASHIO-MIST** using conventional **backpropagation**.  
   * This notebook provides a baseline to compare standard learning methods with bio-inspired learning.  
   * It leverages GPU support for end-to-end training efficiency.

2. **`biolearning.ipynb`**  
   * A reproduction of the **biologically-inspired unsupervised learning** approach from the original paper, followed by a supervised layer.  
   * This notebook showcases the core unsupervised learning technique that drives this entire project, demonstrating how competing hidden units can help the network learn without backpropagation.

3. **`multilayer_bio_learning.ipynb`**  
   * An extended **multi-layer version** of unsupervised learning, where we add depth to the network using multiple hidden layers, followed by a final supervised layer.  
   * This notebook demonstrates the potential of multi-layer architectures in bio-inspired learning setups, and allows comparison to see if deeper unsupervised models yield better representations.

4. **`output`**
   * this directory contains all the results (plots, weights ..)
---

### 🧩 Getting Started

Each notebook is designed to be self-contained, allowing you to jump right into training and experimenting! For best results:

- **Google Colab**: If you want to train models with GPU support, use Google Colab for a seamless experience, especially with `backpropagation.ipynb`.
- **Requirements**: Ensure you have the necessary libraries installed, which are detailed at the start of each notebook.

---

### Let’s Dive In!
I hope you enjoy experimenting with this biologically-inspired extension to traditional neural networks! Dive in, learn, and explore how unsupervised learning and bio-inspired ideas can change the way we think about neural network training.
