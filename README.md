# 🌸 Image Classification with CNN using TensorFlow

This project explores image classification using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras. The dataset contains 3,670 flower images categorized into five classes: **daisy**, **dandelion**, **roses**, **sunflowers**, and **tulips**.

🔗 [Project Notebook](https://github.com/Mukhesh19/Image-Classification)

📦 [Dataset Download](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

---

## 🧠 Objective

Build and evaluate a CNN model to classify images of flowers into their respective categories. The model is trained, validated, and fine-tuned with different hyperparameters, and its strengths and limitations are analyzed.

---

## 📂 Dataset

- **Source**: TensorFlow Example Images
- **Classes**: Daisy, Dandelion, Roses, Sunflowers, Tulips
- **Total Images**: 3,670

---

## ⚙️ Preprocessing

- Resized all images to **180x180**
- Normalized pixel values using a **Rescaling layer**
- Split dataset: **80% Training** / **20% Validation**

---

## 🏗️ Model Architecture

- Built using **Keras Sequential API**
- Layers:
  - `Conv2D` → `MaxPooling2D` (×3)
  - `Flatten` → `Dense` (Fully Connected Layers)
- **Optimizer**: Adam
- **Loss Function**: SparseCategoricalCrossentropy
- **Metrics**: Accuracy

---

## 📊 Training & Evaluation

- Trained for **10 epochs** and later **20 epochs** for fine-tuning
- Evaluation Metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Generated classification report using `sklearn.metrics.classification_report`

---

## 🔍 Results

| Epochs | Training Accuracy | Validation Accuracy | Notes                     |
|--------|--------------------|----------------------|----------------------------|
| 10     | ~98%               | ~63.4%               | Early signs of overfitting |
| 20     | ~99%               | ~67.1%               | Still overfitting observed |

- **Key Insight**: Despite high training accuracy, model fails to generalize due to overfitting.
- **Limitation**: Low precision/recall across classes; potential class imbalance.

---

## 📉 Visualizations

- Training vs. Validation Accuracy and Loss plots
- Classification metrics for each flower class
- Examples of correct and incorrect predictions

---

## 🔁 Future Improvements

- Introduce **data augmentation** to reduce overfitting
- Implement **dropout layers** for regularization
- Experiment with **pre-trained models** (e.g., MobileNet, VGG16)

---

## 📚 References

- Brownlee, J. (2022). [Batch vs Epoch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- GeeksforGeeks (2024). [Epoch in ML](https://www.geeksforgeeks.org/epoch-in-machine-learning/)

---

## 📁 Files Included

- `Image Classification.ipynb` – Jupyter Notebook with full code
- `Image Classification.pdf` – Project report with screenshots and summary

---

## 👋 Author

**Mukhesh Ravi**

Let's connect and collaborate on machine learning and computer vision projects!
