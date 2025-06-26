import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os
from tqdm import tqdm



df = pd.read_csv("data/cleaned_depression_data.csv")
print(df['depression'].value_counts())

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")


class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

#   Train Split 
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['depression'], test_size=0.2, stratify=df['depression'], random_state=42
)

train_dataset = DepressionDataset(train_texts, train_labels, tokenizer)
val_dataset = DepressionDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

#  Model 
class BERTDepressionClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BERTDepressionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bhadresh-savani/bert-base-go-emotion")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        dropout_output = self.dropout(cls_output)
        logits = self.classifier(dropout_output)
        return logits

#  Training Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=df['depression']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model = BERTDepressionClassifier().to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=2e-5)

#  Training Loop 
EPOCHS = 3  

for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    total_train_loss = 0
    correct, total = 0, 0

    # Progress bar for training
    train_bar = tqdm(train_loader, desc="🏋️ Training", leave=False)
    for batch in train_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = correct / total
    print(f" Training Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.4f}")

    #  Validation 
    model.eval()
    all_preds, all_labels = [], []
    val_bar = tqdm(val_loader, desc="🔍 Validating", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(" Validation Results:")
    print(classification_report(all_labels, all_preds, digits=4))


#  Save Model 
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/bert_emotion_model.pt")
tokenizer.save_pretrained("models/bert_emotion_tokenizer.pkl")
print(" Model and tokenizer saved.")
