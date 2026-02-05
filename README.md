# Garbage Classification using CNN (PyTorch)

Garbage classification is an important challenge in modern waste management systems, as improper sorting of waste can cause environmental pollution and reduce recycling efficiency. Traditional manual garbage sorting is time-consuming, labor-intensive, and prone to human error. With increasing waste production in urban areas, automated waste classification systems are becoming essential.

This project implements a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** to classify garbage images into multiple categories. The model is inspired by a **LeNet-based architecture**, designed to demonstrate fundamental deep learning and CNN concepts for image classification.

---

## Project Objectives

### Main Objective
To design and implement a Convolutional Neural Network (CNN) for automatic garbage image classification.

### Specific Objectives
- Preprocess garbage image data for CNN input  
- Design a CNN architecture from scratch using PyTorch  
- Train the CNN model using labeled image data  
- Evaluate model performance using accuracy metrics  

---

## Dataset Description

This project uses the **TrashNet dataset**, containing **2527 images** across **6 garbage categories**:

- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash  

Dataset size: **2527 images**  
Number of classes: **6**

---

## Data Preprocessing

### Image Preparation
- Resized all images to **64 × 64 pixels**
- Converted to **RGB format**
- Normalized pixel values for training stability

### Data Augmentation (Training Only)
To improve generalization, the following augmentations were applied during training:
- Random horizontal flipping
- Random rotation (±15°)
- Color jitter (brightness and contrast variation)

---

## Model Architecture

The CNN model is based on a **LeNet-style design**, consisting of:

- **Input:** 3 × 64 × 64 RGB image  
- **Feature Extraction:**
  - 2 Convolution layers  
  - ReLU activation  
  - MaxPooling layers  
- Flattening layer before classification  
- **Fully Connected Layers:** 256 → 128 → 6  

### Output
- 6 output neurons representing the 6 garbage categories

---

## Training Configuration

- **Framework:** PyTorch  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning Rate:** 0.001  
- **Epochs:** 30  
- **Batch Processing:** Mini-batch DataLoader  
- **Training Method:** Supervised learning with backpropagation  

---

## Model Performance

| Metric           |      Result |
|------------------|-------------|
| Training Accuracy|   **87%**   |
| Testing Accuracy |   **70%**   |
| Total Epochs     |   **30**    |
| Optimizer        |   Adam      |

The results indicate that the model learned useful patterns from the dataset but still faces generalization challenges due to dataset size and class similarity.

---

## Key Achievements

- Successfully implemented a CNN model from scratch using PyTorch  
- Achieved strong training accuracy and reasonable testing performance  
- Demonstrated practical understanding of CNN fundamentals (Conv, Pooling, FC layers)  
- Built a solid baseline model suitable for further improvements and real-world waste classification applications  

---

## Workflow Summary

1. Collected and organized garbage images into class folders  
2. Resized and transformed images into tensor format  
3. Split dataset into training and testing sets  
4. Trained CNN model using supervised learning  
5. Evaluated performance using training and testing accuracy  

---

## How to Run the Project

### Option 1: Run in Jupyter Notebook
Open the notebook file:

-> Garbage.ipynb

### Option 2: Run in Google Colab

Upload the notebook into Google Colab and run all cells.

# Files in this Repository

Garbage-CNN.ipynb : Full implementation (data preprocessing, training, evaluation)

# Future Improvements

  - Apply transfer learning (ResNet, MobileNet, EfficientNet)

  - Hyperparameter tuning for better generalization

  - Add dropout and batch normalization

  - Evaluate with confusion matrix and F1-score

👤 Author

Liza SROY
Data Science Student 

GitHub: https://github.com/LizaSROY

LinkedIn: https://www.linkedin.com/in/lizasroy99/

















