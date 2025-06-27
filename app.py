import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os
import urllib.request
#  Page Config 
st.set_page_config(page_title="MindScan - Depression Check", layout="centered")



def download_model_if_needed():
    model_path = "models/bert_emotion_model.pt"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1vi-SeT_zUUg3so7BomcAwuwKg-hmcng2"
        print(" Downloading model...")
        urllib.request.urlretrieve(url, model_path)
        print(" Model downloaded.")

#  Loading Model and Tokenizer
@st.cache_resource
def load_model():
    
    download_model_if_needed()  # <-- new line

    tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
    bert = AutoModel.from_pretrained("bhadresh-savani/bert-base-go-emotion")

    model = BERTClassifier(bert)
    model.load_state_dict(torch.load("models/bert_emotion_model.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer


class BERTClassifier(torch.nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)

model, tokenizer = load_model()

#  Prediction Helper 
def predict_depression(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = F.softmax(logits, dim=1)
    return probs[0][1].item()

#  Interpretation 
def interpret_emotion(score):
    if score >= 0.8: return "😞 Very low mood and signs of severe distress"
    elif score >= 0.6: return "😕 Slightly low energy, occasional stress"
    elif score >= 0.4: return "🙂 Fairly stable mood with minor concerns"
    else: return "😊 Positive and emotionally balanced"

def interpret_tone(score):
    if score >= 75: return "Distressed and emotionally overwhelmed"
    elif score >= 60: return "Fatigued and struggling emotionally"
    elif score >= 40: return "Some emotional fatigue or uncertainty"
    else: return "Calm and generally stable"

def suggest(score):
    if score >= 75: return "💡 It's strongly recommended you talk to a professional."
    elif score >= 50: return "💬 Some signs of concern. Consider opening up to someone."
    else: return "✅ You seem to be doing well. Stay mindful."

#  Session State Setup 
if "step" not in st.session_state:
    st.session_state.step = 1
if "objective_score" not in st.session_state:
    st.session_state.objective_score = 0
if "open_responses" not in st.session_state:
    st.session_state.open_responses = []

#  UI: Header 
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")
with col2:
    image = Image.open("Data/MIND_SCAN.png")  # Make sure image is in same dir or give full path
    st.image(image, use_column_width=True)
with col3:
    st.write("")

st.markdown("---")

#  Self-Asessment  
if st.session_state.step == 1:
    st.subheader(" Self-Assessment ")

    form = st.form("objective_form")
    score = 0
    responses = []
    questions = [
        " **1.  I often feel sad or down.**",
        " **2.  I have lost interest in activities I used to enjoy.**",
        " **3.  I feel tired or have little energy.**",
        " **4.  I have trouble concentrating.**",
        " **5.  I feel bad about myself.**",
        " **6.  I have difficulty sleeping or sleep too much.**",
        " **7.  I feel hopeless about the future.**",
        " **8.  I feel anxious or on edge frequently.**",
        " **9.  I struggle to find motivation to do everyday tasks.**",
        " **10. I prefer to isolate myself from others.**"
    ]
    scale = {
        "Strongly Disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly Agree": 5
    }

    for q in questions:
        response = form.radio(q, list(scale.keys()), key=q)
        responses.append(scale[response])
        score += scale[response]

    if form.form_submit_button("Next →"):
        st.session_state.objective_score = score
        st.session_state.step = 2
        st.rerun()

#  Self-Reflection
elif st.session_state.step == 2:
    st.subheader("Self-Reflection")

    open_qs = [
        "**1. Describe how you've been feeling emotionally lately.**",
        "**2. Is there something that's been bothering you for a while?**",
        "**3. What do you usually do when you're feeling low?**",
        "**4. Tell me about your sleep and energy levels recently.**",
        "**5. Do you feel hopeful or hopeless about the future?**"
    ]

    form = st.form("open_text_form")
    answers = []
    for i, q in enumerate(open_qs):
        ans = form.text_area(q, key=f"text{i}")
        answers.append(ans.strip())
    if form.form_submit_button("Analyze"):
        st.session_state.open_responses = answers
        st.session_state.step = 3
        st.rerun()

# Output
elif st.session_state.step == 3:
    st.subheader("📊 Final Results")

    obj_score = st.session_state.objective_score
    obj_pct = round((obj_score / 50) * 100)
    mood = interpret_emotion(obj_pct / 100)

    scores = [predict_depression(txt) for txt in st.session_state.open_responses if txt]
    text_avg = sum(scores) / len(scores) if scores else 0
    text_pct = round(text_avg * 100)

    final_score = round((0.7 * obj_pct) + (0.3 * text_pct))
    tone = interpret_tone(final_score)
    advice = suggest(final_score)

    if final_score >= 75: risk = "🔴 High Risk"
    elif final_score >= 50: risk = "🟠 Moderate Risk"
    else: risk = "🟢 Low Risk"

    st.markdown("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    st.markdown("### 🧠 Your Personalized Mental Health Check-In 💬")
    st.markdown("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    st.markdown(f"""
**📋 Self-Assessment (10 Questions):**  
→ Score: **{obj_score}** out of 50  
→ Mood Pattern: {mood}
""")

    st.markdown(f"""
**🧾 AI Reflection (Your responses in your own words):**  
→ Emotional tone: **{tone}**  
→ Depression Indicator Score: **{text_pct} / 100**
""")

    st.markdown(f"""
**🎯 Final Assessment:**  
→ Combined Mental Health Score: **{final_score} / 100**  
→ {risk}
""")

    st.markdown(f"""
**💡 Suggestion:**  
{advice}

You're not alone. You're doing your best, and that matters.
""")
    st.markdown("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
