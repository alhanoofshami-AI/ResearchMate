import os
import tempfile
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. إعداد الصفحة ومفتاح API
st.set_page_config(page_title="ResearchMate", page_icon="📚", layout="wide")
load_dotenv(Path(__file__).resolve().parent / ".env")

# على Streamlit Cloud نقرأ المفتاح من Secrets، ومحليًا نستخدم ملف .env
try:
    cloud_api_key = st.secrets["OPENAI_API_KEY"]
    if cloud_api_key:
        os.environ["OPENAI_API_KEY"] = cloud_api_key
except Exception:
    pass

st.title("📚 ResearchMate")
st.subheader("مساعدك الذكي لتلخيص وتحليل الأوراق العلمية")
st.write("ارفعي ورقتك العلمية بصيغة PDF لتلخيصها وتحليلها وطرح الأسئلة حول محتواها.")


# 2. إعداد النماذج
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# 3. نموذج التحليل المنظم
class ResearchPaper(BaseModel):
    research_topic: str = Field(description="موضوع البحث")
    study_objective: str = Field(description="هدف الدراسة")
    methodology: str = Field(description="المنهجية المستخدمة")
    key_findings: str = Field(description="أهم النتائج")
    conclusion: str = Field(description="الاستنتاجات")


structured_llm = llm.with_structured_output(ResearchPaper)


# 4. قالب التلخيص
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    أنت مساعد أكاديمي متخصص في تلخيص الأوراق العلمية.
    لخّص الورقة باللغة العربية الأكاديمية الواضحة.
    لا تضف معلومات غير موجودة في الورقة.
    """),
    ("human", "لخص الورقة العلمية التالية:\n\n{paper_text}")
])
summary_chain = summary_prompt | llm | StrOutputParser()


# 5. حفظ نتائج الملف داخل جلسة المستخدم
session_defaults = {
    "vectorstore": None,
    "paper_hash": None,
    "summary": None,
    "structured_result": None,
    "page_count": 0
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# 6. رفع ملف PDF
uploaded_file = st.file_uploader("اختاري ملف البحث العلمي", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_file_hash = hashlib.sha256(file_bytes).hexdigest()
    st.success(f"تم اختيار الملف: {uploaded_file.name}")

    if st.button("تلخيص وتحليل البحث العلمي"):
        pdf_path = None
        try:
            with st.spinner("جاري قراءة البحث وبناء قاعدة البحث الخاصة بهذا الملف..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_bytes)
                    pdf_path = temp_file.name

                # قراءة صفحات الملف المرفوع
                documents = PyPDFLoader(pdf_path).load()
                if not documents:
                    st.error("لم أتمكن من استخراج صفحات من ملف PDF.")
                    st.stop()

                paper_text = "\n".join(doc.page_content for doc in documents)
                if not paper_text.strip():
                    st.error("لا يوجد نص قابل للاستخراج. قد يكون الملف عبارة عن صور ممسوحة ضوئيًا.")
                    st.stop()

                # تقسيم الملف الحالي إلى مقاطع
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)

                # إنشاء FAISS من الملف الحالي، بدل تحميل قاعدة ثابتة قديمة
                new_vectorstore = FAISS.from_documents(chunks, embeddings_model)

                summary = summary_chain.invoke({"paper_text": paper_text})
                structured_result = structured_llm.invoke(
                    f"""
                    حلل الورقة العلمية التالية باللغة العربية.
                    استخرج موضوع البحث وهدف الدراسة والمنهجية
                    وأهم النتائج والاستنتاجات.
                    لا تضف معلومات غير موجودة في الورقة.

                    الورقة العلمية:
                    {paper_text}
                    """
                )

                st.session_state.vectorstore = new_vectorstore
                st.session_state.paper_hash = current_file_hash
                st.session_state.summary = summary
                st.session_state.structured_result = structured_result
                st.session_state.page_count = len(documents)

        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

    # لا نعرض نتائج ملف سابق على ملف جديد
    if st.session_state.paper_hash == current_file_hash:
        if st.session_state.summary:
            st.divider()
            st.subheader("📄 الملخص العلمي")
            st.write(st.session_state.summary)

        if st.session_state.structured_result:
            result = st.session_state.structured_result
            st.divider()
            st.subheader("🔍 التحليل المنظم للبحث")

            st.markdown("### 📌 موضوع البحث")
            st.write(result.research_topic)
            st.markdown("### 🎯 هدف الدراسة")
            st.write(result.study_objective)
            st.markdown("### ⚙️ المنهجية")
            st.write(result.methodology)
            st.markdown("### 📊 أهم النتائج")
            st.write(result.key_findings)
            st.markdown("### ✅ الاستنتاجات")
            st.write(result.conclusion)
            st.info(f"عدد صفحات البحث: {st.session_state.page_count}")

        if st.session_state.vectorstore is not None:
            st.divider()
            st.subheader("💬 اسأل بحثك")
            user_question = st.text_input(
                "اكتبي سؤالك عن الورقة العلمية:",
                key="research_question"
            )

            if st.button("إرسال السؤال"):
                if user_question.strip():
                    with st.spinner("جاري البحث عن الإجابة..."):
                        retrieved_docs = st.session_state.vectorstore.similarity_search(
                            user_question, k=3
                        )
                        retrieved_context = "\n\n".join(
                            f"[الصفحة {doc.metadata.get('page', -1) + 1}]\n{doc.page_content}"
                            for doc in retrieved_docs
                        )

                        rag_prompt = ChatPromptTemplate.from_messages([
                            ("system", """
                            أنت ResearchMate، مساعد أكاديمي متخصص.
                            أجب اعتمادًا على السياق المسترجع فقط.
                            لا تضف معلومات من خارج الورقة العلمية.
                            إذا لم تجد الإجابة، وضح أن المعلومات غير متوفرة.
                            اذكر أرقام الصفحات التي استندت إليها.
                            أجب باللغة العربية بوضوح.
                            """),
                            ("human", """
                            السياق:
                            {context}

                            السؤال:
                            {question}
                            """)
                        ])
                        rag_chain = rag_prompt | llm | StrOutputParser()
                        answer = rag_chain.invoke({
                            "context": retrieved_context,
                            "question": user_question
                        })
                        st.markdown("### 📝 الإجابة")
                        st.write(answer)
                else:
                    st.warning("اكتبي السؤال أولًا.")
    else:
        st.info("اضغطي «تلخيص وتحليل البحث العلمي» لمعالجة هذا الملف أولًا.")
else:
    # إفراغ بيانات الملف عند إزالة الملف من أداة الرفع
    for key, value in session_defaults.items():
        st.session_state[key] = value
