import streamlit as st
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx

# --- الإعدادات ---
# الطريقة الصحيحة: قراءة المفتاح من إعدادات Streamlit
#update deplyment
api_key = st.secrets["gsk_AQJscaH31naYkJWCXdqjWGdyb3FYTsKAlt4WfCy7ifiI4G4j3CtlY"]
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


# استخدام Cache لتحميل الموديل مرة واحدة فقط (لزيادة السرعة)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# دالة لاستخراج النص
def extract_text(file):
    try:
        if file.name.endswith('.txt'):
            return file.read().decode("utf-8")
        elif file.name.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in pdf.pages])
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading file: {e}"
    return ""

st.title(" المساعد  الذكي")

# 1. واجهة رفع الملف
uploaded_file = st.file_uploader("قم برفع ملفك (PDF, DOCX, TXT)", type=['txt', 'pdf', 'docx'])

if uploaded_file:
    # معالجة النص
    text = extract_text(uploaded_file)
    documents = [para.strip() for para in text.split('\n\n') if para.strip()]
    
    # بناء الفهرس
    embeddings = model.encode(documents).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    st.success("تم تحليل الملف بنجاح! يمكنك الآن طرح أسئلتك.")

    # 2. منطقة طرح الأسئلة
    question = st.text_input("اطرح سؤالك حول الملف:")
    
    if question:
        # البحث
        distances, indices = index.search(model.encode([question]).astype('float32'), 3)
        context = "\n---\n".join([documents[i] for i in indices[0]])
        
        # التوليد
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "أنت مساعد قانوني خبير. أجب بناءً على السياق المقدم فقط. إذا لم تجد الإجابة، قل لا أعرف."},
                {"role": "user", "content": f"السياق:\n{context}\n\nالسؤال: {question}"}
            ]
        )
        st.write("### الإجابة:")
        st.write(response.choices[0].message.content)
