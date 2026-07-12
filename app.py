import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx

# --- الإعدادات ---
# قراءة المفتاح من إعدادات Streamlit (Secrets)
api_key = st.secrets["GROQ_API_KEY"]
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


# استخدام Cache لتحميل الموديل مرة واحدة فقط (لزيادة السرعة)
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


model = load_model()


def extract_text(uploaded_file):
    """
    تستخرج النص من أي ملف مرفوع وتحوّله إلى نص عادي (str).
    تدعم PDF و DOCX، وأي ملف نصي آخر (txt, md, csv, json, py...).
    """
    if uploaded_file is None:
        return ""

    file_extension = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_extension == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            return "\n".join(pages_text)

        elif file_extension == "docx":
            document = docx.Document(uploaded_file)
            return "\n".join(para.text for para in document.paragraphs)

        else:
            # أي امتداد آخر: نحاول قراءته كنص UTF-8
            raw_bytes = uploaded_file.read()
            try:
                return raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return raw_bytes.decode("utf-8", errors="ignore")

    except Exception as e:
        st.error(f"تعذر قراءة الملف: {e}")
        return ""


st.set_page_config(page_title="المساعد الذكي", page_icon="🤖")
st.title("🤖 المساعد الذكي")

uploaded_file = st.file_uploader(
    "قم برفع ملفك (PDF, DOCX, TXT, MD, CSV, JSON...)",
    type=["txt", "pdf", "docx", "md", "csv", "json", "py"],
)

if uploaded_file:
    # نعالج الملف ونبني الفهرس مرة واحدة فقط، وليس مع كل إعادة تشغيل للصفحة
    file_changed = (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    )

    if file_changed:
        with st.spinner("جاري تحليل الملف وبناء الفهرس..."):
            text = extract_text(uploaded_file)
            documents = [p.strip() for p in text.split("\n\n") if p.strip()]

            if not documents:
                st.error(
                    "لم يتم العثور على نص قابل للقراءة في هذا الملف "
                    "(قد يكون PDF ممسوحاً ضوئياً بدون طبقة نصية)."
                )
                st.stop()

            embeddings = model.encode(documents).astype("float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            st.session_state.file_name = uploaded_file.name
            st.session_state.documents = documents
            st.session_state.index = index

    st.success(
        f"تم تحليل الملف بنجاح! ({len(st.session_state.documents)} مقطع نصي). "
        "يمكنك الآن طرح أسئلتك."
    )

    question = st.text_input("اطرح سؤالك حول الملف:")

    if question:
        with st.spinner("جاري البحث والإجابة..."):
            documents = st.session_state.documents
            index = st.session_state.index

            k = min(3, len(documents))
            q_embedding = model.encode([question]).astype("float32")
            distances, indices = index.search(q_embedding, k)
            context = "\n---\n".join(documents[i] for i in indices[0])

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "أنت مساعد . أجب بناءً على السياق المقدم فقط. "
                            "إذا لم تجد الإجابة، قل لا أعرف."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"السياق:\n{context}\n\nالسؤال: {question}",
                    },
                ],
            )

        st.write("### الإجابة:")
        st.write(response.choices[0].message.content)
else:
    # تنظيف الحالة عند إزالة الملف
    for key in ("file_name", "documents", "index"):
        st.session_state.pop(key, None)
