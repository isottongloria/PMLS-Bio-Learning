<h1 align="center">Physical models of living systems project - PoD<br> University of Padua <br> 2023/2024</h1>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62724611/166108149-7629a341-bbca-4a3e-8195-67f469a0cc08.png" height="120"/>
   
  <img src="https://user-images.githubusercontent.com/62724611/166108076-98afe0b7-802c-4970-a2d5-bbb997da759c.png" height="120"/>
</p>

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

3. **Synaptic Specialization:** I dive into biologically-plausible dynamics by experimenting with networks that have **fixed excitatory and inhibitory synapses** in hidden layers, adding another layer of bio-inspired realism.

---

### 📂 Repository Contents

Here's an overview of the main components in this repository:

1. **`./notebooks/backpropagation.ipynb`**  
   * A Google Colab-ready notebook to train a simple feedforward neural network on **MNIST**, **CIFAR-10**, and **SASHIO-MIST** using conventional **backpropagation**.  
   * This notebook provides a baseline to compare standard learning methods with bio-inspired learning.  
   * It leverages GPU support for end-to-end training efficiency.

2. **`./notebooks/multilayer_bio_learning.ipynb`**
  * A reproduction of the **biologically-inspired unsupervised learning** approach from the original paper, followed by a supervised layer.  
   * An extended **multi-layer version** of unsupervised learning, where we add depth to the network using multiple hidden layers, followed by a final supervised layer.  
   * This notebook demonstrates the potential of multi-layer architectures in bio-inspired learning setups, and allows comparison to see if deeper unsupervised models yield better representations.

3. **`./notebooks/EI_bio_learning.ipynb`**
   * An extended **EI version** of unsupervised learnin
    
4. **`./images`**
   * this directory contains all the important results (plots ..)
---

### 🧩 Getting Started

Each notebook is designed to be self-contained, allowing you to jump right into training and experimenting! For best results:

- **Google Colab**: If you want to train models with GPU support, use Google Colab for a seamless experience, especially with `backpropagation.ipynb`.
- **Requirements**: Ensure you have the necessary libraries installed, which are detailed at the start of each notebook.


### References and important resources

- Krotov & Hopfield (2019) – Unsupervised learning by competing hidden units, PNAS, 116(16), 7723–7731. https://doi.org/10.1073/pnas.1820458116
- Univ. of Padua – Physics of Data MSc 2023/24 – Physical Models of Living Systems – Prof. S. Suweis – Notes by G. Cataldi, F. M.
- Song et al. (2016) – Training EI-RNNs for Cognitive Tasks, PLOS Comput Biol, 12(2): e1004792.
- Dayan (2023) – Hebbian Learning: Biologically Plausible Alternative to Backpropagation, Medium.
- Siri et al. (2007) – Effects of Hebbian learning on the dynamics and structure of random networks with inhibitory and excitatory neurons, Journal of Physiology-Paris, 101(1–3), 136–148.


### Performances and time to run

These simulations were performed on the `NVIDIA GeForce RTX 4050` GPU of my laptop, with each run taking approximately 1 to 5 minutes. The code also runs smoothly on Google Colab. Running on a CPU is possible, but significantly slower.


---

### Let’s Dive In!
I hope you enjoy experimenting with this biologically-inspired extension to traditional neural networks! Dive in, learn, and explore how unsupervised learning and bio-inspired ideas can change the way we think about neural network training.
