# LeNet-5 Paper Reproduction (PyTorch)

[한국어 버전 (Korean Version)](./README_KR.md)

This project is a faithful reproduction of the original **LeNet-5** architecture from the 1998 paper: *"Gradient-Based Learning Applied to Document Recognition"*.

## 🚀 Key Features
Unlike simplified modern versions, this implementation focuses on original details:
* **Trainable Subsampling (S2, S4):** Implemented with learnable weight and bias parameters.
* **C3 Partial Connectivity:** Fully reproduced the sparse connection mapping (Table 1) to break symmetry.
* **Scaled Tanh Activation:** Applied $f(a) = 1.7159 \tanh(\frac{2}{3}a)$ for optimal learning as specified in the paper.
* **MNIST Preprocessing:** Images padded to $32 \times 32$ to match the paper's input specification.

## 📊 Results
* **Accuracy:** 98.6% on MNIST test set (10 Epochs)
* **Final Loss:** 0.0172
* **Tool:** Monitored via TensorBoard (Scalars & Graphs)

## 🛠 Tech Stack
* Python (Anaconda)
* PyTorch, Torchvision
* TensorBoard, tqdm
