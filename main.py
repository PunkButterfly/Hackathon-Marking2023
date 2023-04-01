import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from models.ranking import Ranking
from utils import plot_graph, get_data, get_gtins, get_reg_ids

data = pd.read_csv(r"data\week_closed_gtin.csv")

st.set_page_config(page_title="MARKING HACK", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Название")

navbar_container = st.container()

with navbar_container:
    col1, col2, col3 = st.columns(3)

analytics_tab, ranking_tab, tab3 = st.tabs(["Аналитика", "Рейтинг", "Tab 3"])

with analytics_tab:
    reg_ids = st.selectbox("Регион:", get_reg_ids(data))
    gtins = st.text_input("Товар:", "00C22971781D72C7C475869EC049959A")
    # gtins = st.selectbox("Товар:", "get_gtins()")

    data = get_data(data, gtins, int(reg_ids))
    data = data.sort_values('dt')

    if data is not None:
        if len(data) == 0:
            st.warning('Недостаточно данных')
        else:
            st.dataframe(data)
            graph = plot_graph(data)
            st.plotly_chart(graph)

with ranking_tab:
    ranker = Ranking()
    product_type = st.text_input("Вид товара", value="9199AB529CF62D4BDB7E8B1D7459001D")
    ranked_data = ranker.ranking(product_type)
    if ranked_data is not None:
        if len(ranked_data) == 0:
            st.warning('Недостаточно данных')
        else:
            st.dataframe(ranked_data)

with tab3:
    st.header("Tab 3 Content")
