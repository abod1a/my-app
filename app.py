import time
import os
import streamlit as st

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    APIStatusError,
)

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import PyPDF2
import docx


# --- Settings ---
# Read the API key from Streamlit Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error(
        "⚠️ لم يتم العثور على GROQ_API_KEY في إعدادات Secrets. "
        "أضفه من Manage app → Settings → Secrets."
    )
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

MAX_RETRIES = 4
BASE_DELAY_SECONDS = 2  # doubles on every retry (2s, 4s, 8s, 16s)

# Marker string used to detect an "I don't know" answer from the model
NOT_FOUND_MARKER = "لا أعرف"

# Text splitter: breaks the document into reasonably sized chunks with overlap
# to preserve context continuity, instead of naive splitting on blank lines only
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # size of a single chunk
    chunk_overlap=200,  # overlap to keep context continuity between chunks
    separators=["\n\n", "\n", " ", ""],
)


def ask_model_with_retry(messages, model="llama-3.3-70b-versatile"):
    """
    Sends the request to Groq with automatic retries on rate limiting
    (RateLimitError) or temporary connection issues, using exponential backoff.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages)

        except RateLimitError as e:
            last_error = e
            wait_time = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                st.info(
                    f"⏳ تم تجاوز الحد المسموح من الطلبات، إعادة المحاولة "
                    f"({attempt}/{MAX_RETRIES}) بعد {wait_time} ثانية..."
                )
                time.sleep(wait_time)
            else:
                break

        except APIConnectionError as e:
            last_error = e
            wait_time = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                st.info(f"⏳ مشكلة اتصال مؤقتة، إعادة المحاولة ({attempt}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                break

        except AuthenticationError as e:
            # No point retrying if the key itself is wrong
            st.error("🔑 مفتاح API غير صحيح أو منتهي الصلاحية. تحقق من GROQ_API_KEY في Secrets.")
            st.stop()

        except APIStatusError as e:
            last_error = e
            break  # other server errors (400/500...) are not auto-retried

    # If we get here, all attempts failed
    if isinstance(last_error, RateLimitError):
        st.error(
            "🚦 الخدمة مزدحمة حالياً (تم تجاوز الحد المسموح من الطلبات في Groq). "
            "الرجاء الانتظار دقيقة ثم المحاولة مرة أخرى."
        )
    elif isinstance(last_error, APIConnectionError):
        st.error("🌐 تعذر الاتصال بخادم Groq. تحقق من اتصال الإنترنت وحاول مجدداً.")
    else:
        st.error(f"❌ حدث خطأ غير متوقع أثناء الاتصال بالنموذج: {last_error}")

    return None


def suggest_needed_document(question, model="llama-3.3-70b-versatile"):
    """
    When no answer is found in the current document, we ask the model to
    understand the question's topic and suggest what kind of document the
    user should upload to find the answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "أنت مساعد يحلل أسئلة المستخدمين ليحدد أي نوع من المستندات "
                "قد يحتوي على إجابة لهذا السؤال. لا تجب على السؤال نفسه إطلاقاً، "
                "فقط صف بإيجاز (سطر أو سطرين، بالعربية اذا كان السؤال بالعربي رد بالعربي واذا كان بالانجليزي رد عليه بالانجليزي(الرد يكون باللغة التي طرح بها السؤال)) نوع أو محتوى الملف الذي "
                "من المفترض أن يرفعه المستخدم ليجد فيه إجابة سؤاله. "
                "مثال أسلوب الرد: 'مستند يحتوي على معلومات حول [الموضوع]، "
                "مثل [أمثلة على نوع الملف: عقد، تقرير مالي، سياسة داخلية...]'."
            ),
        },
        {
            "role": "user",
            "content": f"سؤال المستخدم: {question}",
        },
    ]

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception:
        # If this secondary call fails, don't break the main experience for it
        return None


# Disable HF telemetry, and point the HF cache to a writable local folder.
# This matters on constrained/shared environments (e.g. Streamlit Cloud)
# where the default HF cache location may not be writable or may reset.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

# Cache the model with Streamlit so it's loaded only once per session,
# not on every rerun of the script.
@st.cache_resource
def load_model():
    start = time.time()
    # NOTE: correct model repo name is "all-MiniLM-L6-v2" under the
    # "sentence-transformers" namespace. It's small (~80MB) and fast,
    # which makes it a good fit for low-resource machines.
    m = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.environ["HF_HOME"],
    )
    st.write(f"✅ تم تحميل نموذج التضمين في {time.time() - start:.1f} ثانية")
    return m

try:
    with st.spinner("⏳ جاري تحميل نموذج التضمين..."):
        model = load_model()
except Exception as e:
    st.error(f"❌ تعذر تحميل نموذج التضمين (embedding model): {e}")
    st.stop()


def extract_text(uploaded_file):
    """
    Extracts plain text (str) from any uploaded file.
    Supports PDF and DOCX, and falls back to plain-text decoding for any
    other file type (txt, md, csv, json, py...).
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
            # Any other extension: try decoding as UTF-8 text
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
    # Only rebuild the index when the file actually changes, not on every rerun
    file_changed = (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    )

    if file_changed:
        with st.spinner("جاري تحليل الملف وبناء الفهرس..."):
            text = extract_text(uploaded_file)
            documents = [chunk.strip() for chunk in text_splitter.split_text(text) if chunk.strip()]

            if not documents:
                st.error(
                    "لم يتم العثور على نص قابل للقراءة في هذا الملف "
                    "(قد يكون PDF ممسوحاً ضوئياً بدون طبقة نصية)."
                )
                st.stop()

            try:
                embeddings = model.encode(documents).astype("float32")
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء بناء الفهرس: {e}")
                st.stop()

            st.session_state.file_name = uploaded_file.name
            st.session_state.documents = documents
            st.session_state.index = index

    st.success(
        f"تم تحليل الملف بنجاح! ({len(st.session_state.documents)} مقطع نصي). "
        "يمكنك الآن طرح أسئلتك."
    )

    question = st.text_input("اطرح سؤالك حول الملف:")

    if question:
        question = question.strip()

    if question:
        with st.spinner("جاري البحث والإجابة..."):
            documents = st.session_state.documents
            index = st.session_state.index

            try:
                k = min(3, len(documents))
                q_embedding = model.encode([question]).astype("float32")
                distances, indices = index.search(q_embedding, k)

                # Store retrieved chunks so we can reference them as sources later
                st.session_state.retrieved_docs = [documents[i] for i in indices[0]]

                context = "\n---\n".join(st.session_state.retrieved_docs)
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء البحث في الملف: {e}")
                st.stop()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد قانوني خبير. أجب بناءً على السياق المقدم فقط. "
                        f"إذا لم تجد الإجابة في السياق، اكتب حرفياً '{NOT_FOUND_MARKER}' "
                        "في بداية ردك ولا تحاول التخمين أو الإجابة من معلوماتك العامة."
                    ),
                },
                {
                    "role": "user",
                    "content": f"السياق:\n{context}\n\nالسؤال: {question}",
                },
            ]

            response = ask_model_with_retry(messages)

        if response is not None:
            answer = response.choices[0].message.content

            st.write("### الإجابة:")

            if NOT_FOUND_MARKER in answer:
                st.write("🤔 لا أعرف الإجابة بناءً على الملف المرفوع حالياً.")

                with st.spinner("جاري تحديد نوع الملف الذي قد يحتوي على الإجابة..."):
                    suggestion = suggest_needed_document(question)

                if suggestion:
                    st.info(
                        "📄 لمساعدتك في العثور على الإجابة، يبدو أنك بحاجة لرفع:\n\n"
                        f"{suggestion}"
                    )
                else:
                    st.info(
                        "📄 حاول رفع ملف آخر يحتوي على معلومات أقرب لموضوع سؤالك، "
                        "ثم أعد طرح السؤال."
                    )
            else:
                st.write(answer)

                # Show the source chunks the answer was based on
                with st.expander("🔍 عرض النصوص التي تم الاعتماد عليها (المصادر)"):
                    for i, doc in enumerate(st.session_state.retrieved_docs):
                        st.write(f"**المصدر {i+1}:** {doc[:200]}...")
else:
    # Clean up session state when the file is removed
    for key in ("file_name", "documents", "index", "retrieved_docs"):
        st.session_state.pop(key, None)
