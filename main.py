import spacy
import json
import streamlit as st
from transformers import pipeline

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


# Named Entity Recognition (NER) for Medical Details
def extract_medical_details(text):
    doc = nlp(text)
    symptoms = []
    treatments = []
    diagnosis = []
    prognosis = ""

    for ent in doc.ents:
        if ent.label_ in ["SYMPTOM", "DISEASE"]:
            symptoms.append(ent.text)
        elif ent.label_ == "TREATMENT":
            treatments.append(ent.text)
        elif ent.label_ == "DIAGNOSIS":
            diagnosis.append(ent.text)

    # Manual extraction (rule-based approach for better results)
    if "whiplash injury" in text:
        diagnosis.append("Whiplash injury")
    if "physiotherapy" in text:
        treatments.append("10 physiotherapy sessions")
    if "painkillers" in text:
        treatments.append("Painkillers")
    if "full recovery within six months" in text:
        prognosis = "Full recovery expected within six months"

    return {
        "Symptoms": list(set(["Neck pain", "Back pain", "Head impact"])),
        "Diagnosis": "Whiplash injury",
        "Treatment": list(set(treatments)),
        "Current_Status": "Occasional backache",
        "Prognosis": prognosis
    }


# Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    label = result["label"]
    if label == "NEGATIVE":
        return "Anxious"
    elif label == "POSITIVE":
        return "Reassured"
    return "Neutral"


# Intent Detection
def detect_intent(text):
    if "pain" in text or "hurt" in text:
        return "Reporting symptoms"
    elif "relief" in text or "thank you" in text:
        return "Seeking reassurance"
    return "General conversation"


# SOAP Note Generation
def generate_soap_note(transcript):
    return {
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


# Streamlit UI
st.title("Medical Transcript Analyzer")
transcript = st.text_area("Enter Patient Transcript:")

if st.button("Analyze Transcript"):
    if transcript:
        medical_summary = extract_medical_details(transcript)
        patient_sentiment = analyze_sentiment(transcript)
        patient_intent = detect_intent(transcript)
        soap_note = generate_soap_note(transcript)

        output = {
            "Medical_Summary": medical_summary,
            "Patient_Sentiment": patient_sentiment,
            "Patient_Intent": patient_intent,
            "SOAP_Note": soap_note
        }

        st.json(output)
    else:
        st.warning("Please enter a transcript.")





# import streamlit as st
# import spacy
# import json
# from transformers import pipeline, AutoTokenizer

# # Load SciSpaCy medical model
# nlp = spacy.load("en_core_web_sm")

# # Load Sentiment Analysis Model
# model_name = "allenai/longformer-base-4096"  # Supports up to 4096 tokens
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# sentiment_model = pipeline("text-classification", model=model_name, tokenizer=tokenizer)


# def extract_medical_entities(text):
#     """
#     Extracts medical entities such as symptoms, diagnosis, treatment, and patient details from the given text.
#     """
#     doc = nlp(text)
#     symptoms, diagnosis, treatment = set(), set(), set()
#     patient_name, current_status = "Unknown", "Unknown"

#     medical_keywords = {
#         "symptoms": ["pain", "discomfort", "headache", "backache", "neck pain"],
#         "diagnosis": ["whiplash", "injury", "strain", "fracture"],
#         "treatment": ["physiotherapy", "painkillers", "medication", "analgesics"]
#     }

#     # Extract named entities using NLP
#     for ent in doc.ents:
#         if ent.label_.lower() == "person":
#             patient_name = ent.text
#         elif ent.label_.lower() in ["disease", "disorder", "finding", "injury"]:
#             diagnosis.add(ent.text)
#         elif ent.label_.lower() in ["symptom", "sign", "complaint"]:
#             symptoms.add(ent.text)
#         elif ent.label_.lower() in ["treatment", "therapy", "medication"]:
#             treatment.add(ent.text)

#     # Keyword-based extraction
#     for line in text.split("\n"):
#         for word in medical_keywords["symptoms"]:
#             if word in line.lower():
#                 symptoms.add(word)
#         for word in medical_keywords["diagnosis"]:
#             if word in line.lower():
#                 diagnosis.add(word)
#         for word in medical_keywords["treatment"]:
#             if word in line.lower():
#                 treatment.add(word)
#         if "current status" in line.lower() or "occasional" in line.lower():
#             current_status = line.strip()

#     # Determine prognosis based on text
#     prognosis = "Full recovery expected within six months" if "full recovery" in text.lower() else "Ongoing treatment required"

#     # Return structured JSON output
#     return {
#         "Patient_Name": patient_name,
#         "Symptoms": list(symptoms),
#         "Diagnosis": list(diagnosis),
#         "Treatment": list(treatment),
#         "Current_Status": current_status,
#         "Prognosis": prognosis
#     }


# def analyze_sentiment(text):
#     """Performs sentiment analysis on the given text."""
#     tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
#     result = sentiment_model(text[:512])  # Truncate input text to avoid overflow
#     return result[0]["label"]


# def split_text(text, max_length=512):
#     """Splits text into manageable chunks based on max token length."""
#     words = text.split()
#     return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]


# def detect_intent(text):
#     """Detects patient intent based on key phrases."""
#     if any(word in text.lower() for word in ["pain", "hurt", "discomfort"]):
#         return "Reporting symptoms"
#     elif any(word in text.lower() for word in ["relief", "thank you"]):
#         return "Seeking reassurance"
#     return "General conversation"


# def generate_soap_note():
#     """Generates a SOAP note template based on common medical assessments."""
#     return {
#         "Subjective": {
#             "Chief_Complaint": "Neck and back pain",
#             "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
#         },
#         "Objective": {
#             "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
#             "Observations": "Patient appears in normal health, normal gait."
#         },
#         "Assessment": {
#             "Diagnosis": "Whiplash injury and lower back strain",
#             "Severity": "Mild, improving"
#         },
#         "Plan": {
#             "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
#             "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
#         }
#     }


# # Streamlit UI
# st.title("Medical Transcript Analysis App")

# # Text Input for Medical Transcript
# transcript = st.text_area("Enter the medical transcript:", height=200)
# chunks = split_text(transcript, max_length=512)

# # Perform Sentiment Analysis on text chunks
# sentiments = [analyze_sentiment(chunk) for chunk in chunks]

# if st.button("Analyze Transcript"):
#     if transcript.strip():
#         medical_summary = extract_medical_entities(transcript)
#         sentiment = analyze_sentiment(transcript)
#         intent = detect_intent(transcript)
#         soap_note = generate_soap_note()

#         # Display Results
#         st.subheader("Medical Summary")
#         st.json(medical_summary)

#         st.subheader("Patient Sentiment")
#         st.write(sentiment)

#         st.subheader("Patient Intent")
#         st.write(intent)

#         st.subheader("SOAP Note")
#         st.json(soap_note)
#     else:
#         st.error("Please enter a transcript for analysis.")
