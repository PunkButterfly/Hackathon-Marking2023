import streamlit as st

st.set_page_config(page_title="MARKING HACK", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
