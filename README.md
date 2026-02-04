# Gloss-QnA Pipeline

This project performs:
1. Gloss text → English (local model)
2. English → Gemini API (Q/A)
3. Gemini reply → Gloss (local model)

Folder structure:
gloss-qna/
├── data/
│   ├── gloss2sent_train.jsonl
│   └── sent2gloss_train.jsonl
├── models/
│   ├── gloss2sent/
│   └── sent2gloss/
├── src/
│   ├── train_gloss2sent.py
│   ├── train_sent2gloss.py
│   ├── infer.py
│   ├── gemini_client.py
│   └── server.py
├── requirements.txt
└── README.md

Run training:
python src/train_gloss2sent.py
python src/train_sent2gloss.py

Run pipeline:
export GOOGLE_API_KEY="your_key"
python src/infer.py

Start server:
uvicorn src.server:app --reload
