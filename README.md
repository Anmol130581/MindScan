
# 🧠 MindScan

**Prioritize Your Peace of Mind**

MindScan is an interactive, AI-powered Streamlit web app that offers personalized mental health check-ins using both self-assessment and NLP-based emotional analysis. It utilizes a fine-tuned BERT model to estimate depression risk and provides helpful feedback.

---

## 📋 Features

- **Step 1:** Answer 10 objective mental health questions
- **Step 2:** Respond to 5 open-ended emotional check-in prompts
- **Step 3:** Get a final mental health score out of 100 with reflection, emotional tone, risk level, and supportive suggestions
- Uses **BERT-based text classification** for depression risk
- Clean, multi-step user interface with dark-friendly design
- 70-30 weightage system (Objective vs Text-based scores)
- Dynamic emotional feedback (AI reflection based on total score)

---

## 🖼 Interface Preview

![MindScan Logo](Data/MIND_SCAN.png)

---

## 🛠 Tech Stack

- Python
- Streamlit
- HuggingFace Transformers (`AutoTokenizer`, `AutoModel`)
- PyTorch

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MindScan.git
cd MindScan
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
MindScan/
│
├── app.py                     # Main Streamlit app
├── Training.py               # Model training script
├── models/                   # Saved tokenizer, model weights
├── Data/                     # CSV dataset and logo image
├── streamlit/                # Streamlit config files
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 Credits

Developed by Anmol Singh Chauhan  

---

## ⚠ Disclaimer

This app is **not a diagnostic tool**. If you're experiencing symptoms of depression or emotional distress, seek help from a qualified mental health professional.
