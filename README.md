# Text Generation using Deep Learning

This project explores sequence modeling techniques for generating text using Recurrent Neural Networks (RNNs). We implement and compare several neural architectures to produce coherent text based on literary input data.

## 📚 Problem Statement

The objective is to train a model that can generate human-like text character-by-character. Given a corpus of literary text, we aim to predict the next character in a sequence using various deep learning models and ultimately generate readable sequences of text.

## 📁 Dataset

We used a literary dataset composed of text from publicly available sources. The dataset is cleaned and preprocessed into sequences of characters with a fixed length and their corresponding targets (next character).

- Sequence Length: 100 characters
- Vocabulary Size: 58 unique characters

## 🛠️ Tools & Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## 🧪 Model Architectures

We experimented with the following neural network architectures:

1. **Simple RNN**  
2. **Stacked LSTM (2 layers)**  
3. **GRU with Dropout**  
4. **Bidirectional LSTM**

Each model was trained using the Adam optimizer with `categorical_crossentropy` loss function. Early stopping and model checkpointing were used for training stability and performance.

## 🏆 Best Model: Bidirectional LSTM

### 🔧 Architecture & Hyperparameters

- **Embedding Layer**: Output dim = 64
- **Bidirectional LSTM Layer**: 128 units
- **Dropout Layer**: 0.2
- **Dense Layer**: 64 units + ReLU
- **Output Layer**: Softmax with 58 classes (character vocab size)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)

### 📈 Performance

- The Bidirectional LSTM achieved the **lowest training and validation loss** among all architectures.
- Generated text using this model was more coherent, grammatically reasonable, and diverse.
- Sampling techniques like temperature control were employed to balance creativity vs. accuracy in generation.

## ✅ Results

- **Best Model**: Bidirectional LSTM
- **Evaluation Metric**: Categorical accuracy and sample text coherence
- **Sample Outputs**:
