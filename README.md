# 🐶🐱 Dog vs Cat Classification using Transfer Learning

This project uses **Transfer Learning with MobileNetV2** to classify images as **dogs or cats**. The goal is to leverage pre-trained models for accurate and efficient classification, and allow real-time predictions on custom images.

---

## 📂 Dataset

- **Source:** [Dogs vs Cats - Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/)
- **Total Images Used:** 2000 (1000 dogs, 1000 cats)
- **Training Samples:** 1600  
- **Testing Samples:** 400  
- **Image Dimensions:** Resized to 224 × 224 pixels  
- **Channels:** RGB (3 channels)  
- **Labels:** 2 classes (Cat → 0, Dog → 1)

---

## 🚀 Workflow

### 1. **Import Dependencies**
- `NumPy`, `Matplotlib`, `OpenCV`, `PIL`, `os`
- `TensorFlow`, `TensorFlow Hub` for model and transfer learning
- `train_test_split` from `sklearn.model_selection`

### 2. **Data Preprocessing**
- Extract dataset using **Kaggle API**
- Resize images to 224x224 and convert to RGB
- Label images: **Dog = 1**, **Cat = 0**
- Normalize pixel values to [0, 1] range

### 3. **Model Building**
- Use **MobileNetV2** from TensorFlow Hub (as feature extractor)
- Freeze base layers (non-trainable)
- Add final Dense layer with logits (2 units)
- Compile with:
  - **Loss:** Sparse Categorical Crossentropy  
  - **Optimizer:** Adam  
  - **Metric:** Accuracy

### 4. **Train the Model**
- Train for 5 epochs on 1600 images
- Track training accuracy and loss

### 5. **Evaluate Performance**
- Evaluate on 400 test images  
- Display final accuracy and loss

### 6. **Predictive System**
- Upload a custom image
- Resize and normalize
- Predict label using trained model
- Display result: 🐱 or 🐶

---

## 📊 Results

- **Training Accuracy:** ~99.20%  
- **Test Accuracy:** ~98.20%  
- The model performs excellently in distinguishing cats and dogs using a relatively small subset of images.

---

## 📌 Technologies Used

- Python  
- Google Colab / Jupyter Notebook  
- NumPy, Matplotlib  
- OpenCV (for image handling)  
- TensorFlow / Keras / TensorFlow Hub  
- Kaggle API for dataset extraction

---

## 🔑 Key Learnings

- Learned to use **Transfer Learning** to improve accuracy with fewer resources  
- Understood how to preprocess image datasets and normalize RGB data  
- Used **MobileNetV2** as a powerful feature extractor  
- Built a simple but highly accurate classifier using a pre-trained model  
- Created a real-time predictive system for image input

---

## 📥 How to Run

1️⃣ **Clone this repository:**

```bash
git clone https://github.com/RONAKBAGRI/Dog-vs-Cat-Classification-Transfer-Learning.git
```

2️⃣ **Install dependencies:**
```bash
pip install numpy matplotlib opencv-python tensorflow tensorflow-hub kaggle
```

3️⃣ **Run the notebook:**
```bash
jupyter notebook Dog_vs_Cat_Classification_Transfer_Learning.ipynb
```