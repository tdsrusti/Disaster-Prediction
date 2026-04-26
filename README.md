# 🌪️ Disaster Image Classifier

A deep learning web app that classifies disaster images into 4 categories — **Cyclone**, **Earthquake**, **Flood**, and **Wildfire** — using an ensemble of three state-of-the-art CNN models.

---

## 🧠 How It Works

Three backbone models each extract 256 features from an input image. These 768 combined features are passed to a dense meta-classifier that outputs the final prediction.

```
Image → EfficientNetV2  ─┐
Image → ConvNeXt        ─┼─► [256+256+256 = 768 features] ─► Dense Classifier ─► Prediction
Image → DenseNet201     ─┘
```

---

## 🗂️ Project Structure

```
disaster-project/
├── app.py                        # Flask backend (REST API)
├── index.html                    # Frontend UI
├── requirements.txt              # Python dependencies
├── class_names.json              # Class labels
├── EfficientNetV2_final.keras    # Backbone 1 (not in repo — too large)
├── ConvNeXt_final.keras          # Backbone 2 (not in repo — too large)
├── DenseNet201_final.keras       # Backbone 3 (not in repo — too large)
└── dense_classifier_final.keras  # Meta-classifier (not in repo — too large)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/disaster-classifier.git
cd disaster-classifier
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Download model files
The `.keras` model files are not included in this repo due to their size.
Download them from Google Drive and place them in the project root:
- `EfficientNetV2_final.keras`
- `ConvNeXt_final.keras`
- `DenseNet201_final.keras`
- `dense_classifier_final.keras`

### 4. Run the backend
```bash
python3 app.py
```
The server starts at `http://localhost:5001`

### 5. Open the frontend
Open `index.html` in your browser (or use VS Code Live Server).

---

## 🚀 API Usage

### Health check
```
GET http://localhost:5001/
```
Response:
```json
{"status": "running", "classes": ["cyclone", "earthquake", "flood", "wildfire"]}
```

### Predict
```
POST http://localhost:5001/predict
Content-Type: multipart/form-data
Body: file=<image>
```
Response:
```json
{
  "predicted_class": "wildfire",
  "confidence": 0.97,
  "all_probabilities": {
    "cyclone": 0.01,
    "earthquake": 0.01,
    "flood": 0.01,
    "wildfire": 0.97
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, TensorFlow / Keras |
| Models | EfficientNetV2, ConvNeXt, DenseNet201 |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Training | Google Colab (Keras 3.13.2) |

---

## 📋 Requirements

- Python 3.10+
- TensorFlow 2.20+
- Keras 3.13.2
- See `requirements.txt` for full list

---

## 📌 Notes

- Port 5000 is used by AirPlay on macOS — the app runs on **port 5001** instead
- Model files must be in `.keras` format (not `.h5`) for compatibility
- All protobuf warnings on startup are harmless and can be ignored
