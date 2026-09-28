from pathlib import Path

import streamlit as st

from docqa import AppError, answer_question, build_index, load_index


HOME = Path(__file__).resolve().parent
DOCUMENTS = HOME / "documents"
INDEX = HOME / "data" / "index.json"
REPO = "https://github.com/balyakin/local-docs-ai"

st.set_page_config(page_title="local-docs-ai", page_icon="📄", layout="wide")
st.image(HOME / "assets" / "cover.svg", width="stretch")
badges = st.columns(3)
with badges[0]:
    st.link_button("↗ Быстрый запуск", f"{REPO}#быстрый-запуск", width="stretch")
with badges[1]:
    st.link_button("↗ Установка Ollama", f"{REPO}/blob/main/docs/ollama.md", width="stretch")
with badges[2]:
    st.link_button("↗ Семь уроков", f"{REPO}/blob/main/lessons/01-start.md", width="stretch")
st.caption("На обложке — пример из учебного файла. Ваш ответ появится ниже.")
st.subheader("Задайте вопрос")
st.write("Положите файлы в папку `documents` и обновите индекс. Под ответом будут фрагменты, по которым его можно проверить.")

with st.sidebar:
    st.header("Ваши документы")
    st.code(str(DOCUMENTS), language=None)
    st.caption("Подходят файлы .md и .txt в UTF-8. До 50 файлов, вместе не больше 1 МБ.")
    if st.button("Обновить индекс", width="stretch"):
        st.session_state.pop("result", None)
        with st.spinner("Читаю документы и строю индекс…"):
            try:
                index = build_index(DOCUMENTS, INDEX)
            except AppError as exc:
                st.error(str(exc))
            else:
                st.success(f"Готово: {len(index['chunks'])} фрагментов.")
    try:
        index = load_index(DOCUMENTS, INDEX)
    except AppError as exc:
        st.info(str(exc))
    else:
        st.caption(f"В индексе {len(index['chunks'])} фрагментов.")
    st.link_button("Как установить Ollama", f"{REPO}/blob/main/docs/ollama.md", width="stretch")

with st.form("question_form"):
    question = st.text_input("Вопрос", placeholder="Когда можно забрать велосипед?")
    submitted = st.form_submit_button("Спросить")

if submitted:
    st.session_state.pop("result", None)
    with st.spinner("Ищу ответ…"):
        try:
            st.session_state.result = answer_question(question, DOCUMENTS, INDEX)
        except AppError as exc:
            st.error(str(exc))

if "result" in st.session_state:
    answer, sources = st.session_state.result
    st.subheader("Ответ")
    st.write(answer)
    st.subheader("Фрагменты для проверки")
    for number, source in enumerate(sources, 1):
        label = f"[{number}] {source['path']}, строки {source['start_line']}–{source['end_line']}"
        with st.expander(label):
            st.code(source["text"], language=None)
    st.caption("Модель может ошибиться. Проверьте ответ по показанным фрагментам.")
