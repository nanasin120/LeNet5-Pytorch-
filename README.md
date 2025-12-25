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

##🛠 Installation & Usage
1. Environment Setup
Option A: Using Conda (Recommended)
You can replicate the exact development environment used in this project using the provided environment.yml file.

```
# Create the environment (the name is defined within environment.yml)

conda env create -f environment.yml

# Activate the environment

conda activate <environment_name>
```
Option B: Using Pip
Alternatively, you can install the required dependencies directly using pip.
```
pip install -r requirements.txt
```
2. Training the Model
Run the following command to start the training process on the MNIST dataset. The script will automatically download the dataset if it's not present.
```
python train.py
```
3. Monitoring Results
You can monitor the training progress, including loss/accuracy curves and the model's computational graph, using TensorBoard.
```
tensorboard --logdir=runs
```
Once the server is running, open your web browser and navigate to: http://localhost:6006
