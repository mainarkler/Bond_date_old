from __future__ import annotations

import streamlit as st

from services.emission_documents import (
    analyze_document,
    build_analysis_report,
    extract_document,
)

st.set_page_config(
    page_title="Анализ эмиссионных документов",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Анализ эмиссионного документа")
st.caption("Отдельное приложение: PDF/DOCX → OCR → поиск важных условий → Qwen3 → подробный PDF-отчёт")

if "document" not in st.session_state:
    st.session_state.document = None
if "result" not in st.session_state:
    st.session_state.result = None
if "report" not in st.session_state:
    st.session_state.report = None
if "signature" not in st.session_state:
    st.session_state.signature = None

uploaded = st.file_uploader(
    "Загрузите один эмиссионный документ",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=False,
)

if uploaded is not None:
    signature = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.signature != signature:
        st.session_state.signature = signature
        st.session_state.document = None
        st.session_state.result = None
        st.session_state.report = None

    if st.button("1️⃣ Распознать текст", type="primary", use_container_width=True):
        with st.status("Распознаём документ…", expanded=True) as status:
            try:
                document = extract_document(uploaded.getvalue(), uploaded.name)
                st.session_state.document = document
                st.session_state.result = None
                st.session_state.report = None
                methods = {}
                for page in document.pages:
                    methods[page.method] = methods.get(page.method, 0) + 1
                st.write(f"Страниц: {len(document.pages)}")
                st.write(f"Методы: {methods}")
                status.update(label="Текст документа готов", state="complete")
            except Exception as exc:
                status.update(label="Ошибка распознавания", state="error")
                st.error(str(exc))

document = st.session_state.document

if document is not None:
    st.success(f"Документ распознан: {document.filename}, страниц — {len(document.pages)}")

    with st.expander("Показать распознанный текст"):
        st.text(document.text[:50000])

    if st.button("2️⃣ Найти важное и проанализировать", type="primary", use_container_width=True):
        progress = st.empty()
        try:
            result = analyze_document(document, progress=progress.info)
            st.session_state.result = result
            st.session_state.report = build_analysis_report(document, result)
            progress.success("Анализ завершён.")
        except Exception as exc:
            progress.error(f"Ошибка анализа: {exc}")

result = st.session_state.result

if result is not None:
    facts = result.get("facts", {})
    analysis = result.get("analysis", {})

    st.subheader("Краткий вывод")
    st.write(analysis.get("summary", "не указано в документе"))

    st.subheader("Ключевые условия")
    for key in ("issuer", "security_type", "issue_volume", "nominal", "currency",
                "maturity", "placement_period", "coupon", "amortization"):
        st.write(f"**{key}:** {facts.get(key, 'не указано в документе')}")

    st.subheader("Важные условия и внимание")
    for item in analysis.get("attention_points", []):
        st.warning(str(item))

    st.subheader("Риски")
    for item in analysis.get("risks", []):
        st.write(f"• {item}")

    with st.expander("Найденные важные фрагменты"):
        for hit in result.get("keyword_hits", []):
            st.markdown(
                f"**Стр. {hit.get('page', '?')} — {hit.get('keyword', '')}**  
"
                f"{hit.get('fragment', '')}"
            )

    report = st.session_state.report
    if report:
        st.download_button(
            "📥 Скачать подробный PDF-отчёт",
            data=report,
            file_name="analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
