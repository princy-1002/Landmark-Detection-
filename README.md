# Famous Landmark Detection using Deep Learning

An image classification project that recognizes famous landmarks/monuments (e.g. Eiffel Tower, Taj Mahal, Statue of Liberty, Colosseum, Great Wall of China) from photographs, using **Transfer Learning** with a pretrained **MobileNetV2** CNN.

> Built as an internship project. End-to-end pipeline from data loading to inference.

---

## Project Pipeline

1. Setup & Imports
2. Dataset Acquisition
3. Data Exploration & Visualization
4. Data Preprocessing & Augmentation
5. Model Building (Transfer Learning)
6. Model Training (head training + fine-tuning)
7. Training Results Visualization
8. Model Evaluation (Confusion Matrix, Classification Report)
9. Save the Trained Model
10. Inference on New / Custom Images
11. Conclusion & Future Work

---

## Repo Structure

```
landmark-detection/
├── landmark_detection_project.ipynb   # main notebook
├── data/                                # dataset (not tracked in git)
│   ├── train/
│   │   ├── eiffel_tower/
│   │   ├── taj_mahal/
│   │   ├── statue_of_liberty/
│   │   ├── colosseum/
│   │   └── great_wall_of_china/
│   └── test/
│       └── ... (same class folders)
├── models/                               # saved trained models
│   └── landmark_classifier.keras
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/landmark-detection.git
cd landmark-detection
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the dataset
Download a landmark image dataset (Kaggle's "Landmark Classification", "Famous Landmarks around the World", or a subset of Google Landmarks Dataset v2 work well) and arrange it as:

```
data/train/<class_name>/*.jpg
data/test/<class_name>/*.jpg
```

### 4. Run the notebook
```bash
jupyter notebook landmark_detection_project.ipynb
```

---

## Model

- **Backbone:** MobileNetV2 (pretrained on ImageNet, `include_top=False`)
- **Training strategy:** Two-phase transfer learning
  - Phase 1 — train a new classification head with the backbone frozen
  - Phase 2 — unfreeze the top ~30 layers of the backbone and fine-tune at a low learning rate
- **Callbacks:** `EarlyStopping` and `ReduceLROnPlateau` to avoid overfitting

## Results

_Add your final accuracy, confusion matrix, and classification report here after training._

| Metric | Value |
|---|---|
| Train Accuracy | — |
| Validation Accuracy | — |
| Test Accuracy | — |

## Future Work

- Try other backbones (EfficientNetB0, ResNet50) and compare accuracy
- Add Grad-CAM visualizations to show which part of the image the model focused on
- Deploy as a Streamlit/Flask web app for photo uploads
- Expand dataset with more landmark classes and images per class
- Add a script to responsibly scrape/download landmark images to grow the dataset

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
