# Diabetic Retinopathy Classification using Deep Learning

## Overview

This project presents a deep learning-based automated system for diabetic retinopathy (DR) severity classification using EfficientNet-B0 architecture. The system analyzes retinal fundus photographs to classify DR severity across five categories, achieving 76.47% validation accuracy with substantial clinical agreement (κ = 0.6279).


While the model shows excellent performance on common cases (No_DR: 97% F1-score), it has critical limitations in detecting severe cases that require immediate medical attention. The system requires significant improvements before clinical deployment.

## 📊 Dataset

- **Total Images**: 3,662 preprocessed retinal fundus photographs
- **Training Set**: 2,931 images (80.04%)
- **Validation Set**: 731 images (19.96%)
- **Classes**: 5 severity levels of diabetic retinopathy

### Class Distribution

| Class | Training Count | Validation Count | Total | Percentage |
|-------|----------------|------------------|-------|------------|
| No_DR | 1,800 | 361 | 2,161 | 59.02% |
| Moderate | 994 | 199 | 1,193 | 32.58% |
| Mild | 365 | 74 | 439 | 11.99% |
| Severe | 188 | 38 | 226 | 6.17% |
| Proliferate_DR | 290 | 59 | 349 | 9.53% |

## 🏗️ Model Architecture

The system employs a transfer learning approach with EfficientNet-B0:

```
Input Layer (224×224×3)
↓
EfficientNet-B0 Backbone (ImageNet pretrained)
↓
GlobalAveragePooling2D
↓
Dropout (rate=0.2)
↓
Dense(128) + ReLU + Dropout(0.2)
↓
Dense(5) + Softmax
```

### Model Specifications
- **Base Model**: EfficientNet-B0 (ImageNet pretrained)
- **Total Parameters**: 4.21M (4.00M trainable, 0.21M frozen)
- **Optimizer**: Adam (learning rate: 1×10⁻⁴)
- **Loss Function**: Categorical Cross-entropy
- **Batch Size**: 32
- **Max Epochs**: 45

## 📈 Performance Metrics

### Overall Performance
- **Accuracy**: 76.47%
- **Weighted Precision**: 69.66%
- **Weighted Recall**: 76.47%
- **Weighted F1-Score**: 72.08%
- **Cohen's Kappa**: 0.6279 (Substantial agreement)

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No_DR | 0.95 | 0.98 | 0.97 | 361 |
| Moderate | 0.59 | 0.87 | 0.70 | 199 |
| Mild | 0.55 | 0.42 | 0.48 | 74 |
| Severe | 0.17 | 0.05 | 0.08 | 38 |
| Proliferate_DR | 0.00 | 0.00 | 0.00 | 59 |

## 🔧 Installation

### Requirements
```bash
pip install tensorflow>=2.8.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install scikit-learn>=1.0.0
pip install opencv-python>=4.5.0
```

### Setup
```bash
git clone https://github.com/Tech-Savant20/iabetic-Retinopathy-Detection.git
cd iabetic-Retinopathy-Detection
pip install -r requirements.txt
```

## 🚀 Usage

### Training
```python
from model import create_model
from data_loader import load_data

# Load and preprocess data
train_data, val_data = load_data('gaussian_filtered_images/')

# Create and compile model
model = create_model()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=45,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)
```

### Inference
```python
from model import load_trained_model
from preprocessing import preprocess_image

# Load trained model
model = load_trained_model('best_model.h5')

# Preprocess image
image = preprocess_image('path/to/retinal_image.jpg')

# Make prediction
prediction = model.predict(image)
severity_class = get_severity_class(prediction)
```

## 📁 Project Structure

```
diabetic-retinopathy-classification/
├── data/
│   ├── gaussian_filtered_images/
│   ├── train/
│   └── validation/
├── models/
│   ├── efficientnet_model.py
│   └── best_model.h5
├── preprocessing/
│   ├── data_loader.py
│   └── augmentation.py
├── evaluation/
│   ├── metrics.py
│   └── confusion_matrix.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_training.ipynb
├── requirements.txt
└── README.md
```

## ⚠️ Critical Limitations

### Severe Case Detection Issues
- **Severe Class Recall**: Only 5.3% of severe cases correctly identified
- **Proliferative DR Detection**: 0% detection rate (critical safety issue)
- **Systematic Bias**: 86.8% of severe cases misclassified as moderate

### Clinical Risk Assessment
| Risk Level | Cases | Model Performance | Clinical Recommendation |
|------------|-------|-------------------|------------------------|
| Low Risk (No_DR) | 361 | 97.8% accurate | Suitable for screening |
| Moderate Risk (Mild/Moderate) | 273 | 69.6% accurate | Requires review |
| High Risk (Severe/Proliferate) | 97 | 2.1% accurate | **Not suitable for clinical use** |

## 🔮 Future Improvements

### Immediate Priority (1-3 months)
- [ ] Implement focal loss for class imbalance
- [ ] Increase input resolution to 380×380
- [ ] Add advanced data augmentation (Mixup, CutMix)
- [ ] Implement ensemble of multiple models
- [ ] Hyperparameter optimization

### Medium-term Goals (3-6 months)
- [ ] Collect additional training data for rare classes
- [ ] Implement curriculum learning strategy
- [ ] Add model interpretability (GradCAM)
- [ ] Conduct clinical validation study
- [ ] Optimize for real-time inference

### Long-term Goals (6-12 months)
- [ ] Integration with clinical workflow systems
- [ ] Multi-center validation studies
- [ ] Regulatory compliance assessment
- [ ] Production deployment
- [ ] Continuous learning system

## 📚 Technical Specifications

- **Framework**: TensorFlow/Keras
- **Hardware**: GPU-accelerated training recommended
- **Training Time**: ~45 epochs
- **Inference Time**: <1 second per image
- **Model Size**: 4.21M parameters


### Priority Areas for Contribution
1. Class imbalance mitigation techniques
2. Advanced data augmentation methods
3. Model interpretability features
4. Clinical validation protocols
5. Production optimization


## 🔬 Citation

If you use this work in your research, please cite:

```bibtex
@misc{diabetic_retinopathy_classification_2023,
  title={Deep Learning-Based Automated Diabetic Retinopathy Severity Assessment},
  author={Tech-Savant20},
  year={2023},
  howpublished={GitHub Repository},
  url={https://github.com/Tech-Savant20/iabetic-Retinopathy-Detection}
}
```


## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/Tech-Savant20/iabetic-Retinopathy-Detection.git

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --data_path gaussian_filtered_images/ --epochs 45

# Evaluate model
python evaluate.py --model_path best_model.h5 --test_data validation/
```

---![screencapture-colab-research-google-drive-1SHeOOxywvcvjOOl2sUUJi7PMvFdr5kwS-2025-07-06-00_55_02](https://github.com/user-attachments/assets/398968ce-ce3f-4613-add9-ddaf72ee0c61)



