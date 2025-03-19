# Medical NLP Pipeline

## Overview
This project is an **AI-based medical transcription and analysis pipeline** that processes physician-patient conversations. The system extracts key medical details, performs sentiment and intent analysis, and generates SOAP notes.

## Features
- **Named Entity Recognition (NER):** Extracts symptoms, treatments, diagnoses, and prognosis.
- **Text Summarization:** Converts medical transcripts into structured reports.
- **Sentiment & Intent Analysis:** Classifies patient concerns as `Anxious`, `Neutral`, or `Reassured`.
- **SOAP Note Generation:** Creates structured clinical notes for medical documentation.

## Installation & Setup

### Clone the Repository:
```sh
git clone https://github.com/your-repo/medical-nlp-pipeline.git
cd medical-nlp-pipeline
```

### Create a Virtual Environment:
```sh
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### Install Dependencies:
```sh
pip install -r requirements.txt
```

### Run the Script:
```sh
python main.py
```

## Expected Output (JSON Format)
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months",
  "Sentiment_Analysis": {
    "Sentiment": "Reassured",
    "Intent": "Seeking reassurance"
  },
  "SOAP_Note": {
    "Subjective": {
      "Chief_Complaint": "Neck and back pain",
      "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
    },
    "Objective": {
      "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
      "Observations": "Patient appears in normal health, normal gait."
    },
    "Assessment": {
      "Diagnosis": "Whiplash injury and lower back strain",
      "Severity": "Mild, improving"
    },
    "Plan": {
      "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
      "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
    }
  }
}
```

## Q&A Section

### How would you handle ambiguous or missing medical data in the transcript?
- Use **context-based inference** to predict missing details.
- Utilize **pre-trained medical NER models** like `scispaCy` for enhanced entity recognition.
- Implement **default categories** (e.g., 'Unknown diagnosis') for missing values.

### What pre-trained NLP models would you use for medical summarization?
- `BART` and `T5` are effective for medical text summarization.
- `BioBERT` or `ClinicalBERT` for domain-specific improvements.

### How would you fine-tune BERT for medical sentiment detection?
- **Dataset:** Use `MedQuad`, `MIMIC-III`, or `NRC Emotion Lexicon`.
- **Training:** Fine-tune `BioBERT` or `DistilBERT` using medical dialogue datasets.

### What datasets would you use for training a healthcare-specific sentiment model?
- `MIMIC-III` (Clinical notes dataset).
- `i2b2` dataset (Annotated medical text for NLP tasks).
- `Physician-patient dialogues dataset` for real-world sentiment examples.

### How would you train an NLP model to map transcripts into SOAP format?
- Use **sequence-to-sequence models** (`T5`, `GPT-4`) trained on SOAP note datasets.
- Apply **rule-based heuristics** for structured section generation.
- Leverage **weakly supervised learning** using partially labeled SOAP notes.

---

Feel free to modify or expand upon this README as needed! 🚀

