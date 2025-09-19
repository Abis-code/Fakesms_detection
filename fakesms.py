# app.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ---- Model Definition ----
class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=128, num_classes=2):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = lstm_out[:, -1, :]  # last hidden state
        out = self.dropout(lstm_out)
        return self.fc(out)

# ---- Load Tokenizer & Model ----
tokenizer = BertTokenizer.from_pretrained("./saved_tokenizer")
model = BertLSTMClassifier(bert_model_name="bert-base-uncased")

# Load trained weights
state_dict = torch.load("bert_lstm_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---- Streamlit UI ----
st.title("Fake SMS Detection")

user_input = st.text_area("✉️ Enter an SMS message:", "")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning(" Please enter a message first.")
    else:
        # Tokenize input
        encoding = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().item()

        # Map prediction to label
        label_map = {0: " This SMS is NOT Spam", 1: " This SMS is SPAM"}
        st.subheader(label_map[pred])
        st.write(f"Confidence: {probs[0][pred].item():.2f}")
