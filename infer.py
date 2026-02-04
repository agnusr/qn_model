# src/infer.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
from qn_client import ask_gemini
import re
import torch

print("✅ Loading models...")

# ✅ Load models
g2s_tokenizer = T5Tokenizer.from_pretrained("models/gloss2sent")
g2s_model = T5ForConditionalGeneration.from_pretrained("models/gloss2sent")

s2g_tokenizer = T5Tokenizer.from_pretrained("models/sent2gloss")
s2g_model = T5ForConditionalGeneration.from_pretrained("models/sent2gloss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g2s_model.to(device)
s2g_model.to(device)

# ✅ TASK PREFIXES (VERY IMPORTANT)
G2S_PREFIX = "translate Gloss to English: "
S2G_PREFIX = "translate English to Gloss: "


# ✅ Proper generation function
def generate(model, tokenizer, text, task_prefix=""):
    full_text = task_prefix + text.strip()

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_length=64,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ✅ Clean Gemini output before Gloss conversion
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[*:•-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ✅ -----------------------------
# ✅ MAIN INTERACTIVE LOOP
# ✅ -----------------------------
while True:
    gloss_input = input("\n👉 Enter Gloss Text (or type exit): ").strip().upper()

    if gloss_input.lower() == "exit":
        break

    # ✅ 1️⃣ Gloss → English
    english = generate(
        g2s_model,
        g2s_tokenizer,
        gloss_input,
        task_prefix=G2S_PREFIX
    )

    if not english.strip():
        print("⚠️ Gloss → English model failed.")
        continue

    print("✅ Gloss Input :", gloss_input)
    print("✅ English Output :", english)

    # ✅ 2️⃣ English → Gemini
    prompt = f"""
Give ONLY short factual answer.
No bullet points.
No explanations.
No multiple options.
Make sure the answer is worded in simple and clear language.
It must be a single sentence.
Make sure the answer is relevant to the question.
Small and concise answers are preferred.
It must be easily convertible to Sign Language Gloss.

Question: {english}
"""
    gemini_answer = ask_gemini(prompt)

    if gemini_answer == "OFFLINE":
        print("⚠️ Offline → Using English as Answer")
        gemini_answer = english

    print("✅ Answer :", gemini_answer)

    # ✅ 3️⃣ English → Gloss
    cleaned_answer = clean_text(gemini_answer)

    final_gloss = generate(
        s2g_model,
        s2g_tokenizer,
        cleaned_answer,
        task_prefix=S2G_PREFIX
    )

    print("✅ Final Gloss Output :", final_gloss)