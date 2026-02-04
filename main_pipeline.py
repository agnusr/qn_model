# src/main_pipeline.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
from gemini_api import ask_gemini

# ✅ Load models
gloss2sent_tokenizer = T5Tokenizer.from_pretrained("gloss2sent_model")
gloss2sent_model = T5ForConditionalGeneration.from_pretrained("gloss2sent_model")

sent2gloss_tokenizer = T5Tokenizer.from_pretrained("sent2gloss_model")
sent2gloss_model = T5ForConditionalGeneration.from_pretrained("sent2gloss_model")


# ✅ GLOSS → ENGLISH
def gloss_to_english(gloss_text):
    inputs = gloss2sent_tokenizer(gloss_text, return_tensors="pt")
    outputs = gloss2sent_model.generate(**inputs)
    return gloss2sent_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ✅ ENGLISH → GLOSS
def english_to_gloss(sentence):
    inputs = sent2gloss_tokenizer(sentence, return_tensors="pt")
    outputs = sent2gloss_model.generate(**inputs)
    return sent2gloss_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ✅ FULL PIPELINE
while True:
    gloss = input("\nENTER GLOSS: ")

    english = gloss_to_english(gloss)
    print("ENGLISH:", english)

    answer = ask_gemini(english)
    print("GEMINI:", answer)

    final_gloss = english_to_gloss(answer)
    print("FINAL GLOSS:", final_gloss)


