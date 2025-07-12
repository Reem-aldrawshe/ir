import streamlit as st
import requests

st.set_page_config(page_title="IR System", page_icon="🔍")

st.title("🔍 Information Retrieval System")

st.markdown("### أدخل استعلام البحث:")

query = st.text_input("Query", "")

col1, col2 = st.columns(2)

with col1:
    dataset = st.selectbox(
        "Dataset",
        ["antique", "beir"]
    )

with col2:
    method = st.selectbox(
        "Search Method",
        ["bert", "tfidf", "hybrid"]
    )

top_k = st.number_input(
    "Number of results (top_k)",
    min_value=1,
    max_value=100,
    value=10
)

if st.button("🔍 Search"):
    if not query.strip():
        st.error("يرجى إدخال استعلام أولاً.")
    else:
        with st.spinner("جاري البحث..."):
            res = requests.post(
                "http://127.0.0.1:8001/search",
                json={
                    "query": query,
                    "dataset": dataset,
                    "method": method,
                    "top_k": top_k
                }
            )

            if res.status_code == 200:
                results = res.json()["results"]
                st.success(f"عدد النتائج: {len(results)}")
                for i, r in enumerate(results, 1):
                    st.markdown(f"""
                    **{i}. Doc ID: {r['doc_id']}**
                    - Original: {r['original_text']}
                    - Clean: {r['clean_text']}
                    - Score: `{r['score']}`
                    """)
            else:
                st.error(f"خطأ في البحث: {res.status_code}")

st.markdown("---")

if st.button("⭐ Go to Evaluation"):
    st.info("🔗 تمهيد للذهاب إلى صفحة التقييم...")
    st.write("(سنربطها لاحقاً مع صفحة تقييم منفصلة)")




# import streamlit as st
# import requests

# st.set_page_config(page_title="IR System", page_icon="🔍", layout="wide")

# # 🎨 ألوان
# st.markdown(
#     """
#     <style>
#     .main {background-color: #f7fbfd;}
#     .stButton>button {
#         background-color: #007acc;
#         color: white;
#         border-radius: 8px;
#         padding: 6px 12px;
#         font-weight: bold;
#     }
#     .stButton>button:hover {
#         background-color: #005b99;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ⭐ زر التقييم بالأعلى
# col_e1, col_e2 = st.columns([1, 6])
# with col_e1:
#     if st.button("⭐ تقييم", use_container_width=True):
#         st.info("🔗 جاري التحويل إلى صفحة التقييم...")
#         st.stop()

# with col_e2:
#     st.title("🔍 Information Retrieval System")

# st.markdown("### أدخل استعلام البحث:")

# query = st.text_input("Query", "")

# col1, col2 = st.columns(2)

# with col1:
#     dataset = st.selectbox(
#         "Dataset",
#         ["antique", "beir"]
#     )

# with col2:
#     method = st.selectbox(
#         "Search Method",
#         ["bert", "tfidf", "hybrid"]
#     )

# top_k = st.slider(
#     "عدد النتائج",
#     min_value=1,
#     max_value=100,
#     value=10
# )

# if st.button("🔍 Search"):
#     if not query.strip():
#         st.error("يرجى إدخال استعلام أولاً.")
#     else:
#         with st.spinner("جاري البحث..."):
#             res = requests.post(
#                 "http://127.0.0.1:8001/search",
#                 json={
#                     "query": query,
#                     "dataset": dataset,
#                     "method": method,
#                     "top_k": top_k
#                 }
#             )

#             if res.status_code == 200:
#                 results = res.json()["results"]
#                 st.success(f"✅ عدد النتائج: {len(results)}")
#                 for i, r in enumerate(results, 1):
#                     st.markdown(f"""
#                     <div style="background-color:#eaf4fc; padding:10px; margin-bottom:10px; border-radius:8px;">
#                     <b>{i}. Doc ID:</b> {r['doc_id']}<br>
#                     <b>Original:</b> {r['original_text']}<br>
#                     <b>Clean:</b> <i>{r['clean_text']}</i><br>
#                     <b>Score:</b> <code>{r['score']}</code>
#                     </div>
#                     """, unsafe_allow_html=True)
#             else:
#                 st.error(f"خطأ في البحث: {res.status_code}")
