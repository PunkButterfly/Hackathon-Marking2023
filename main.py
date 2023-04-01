import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.express as px
from models.ranking import Ranking


def plot_graph(df, x_axis, y_axis):
    fig = px.line(df, x=data['dt'], y=data['cnt'], title="График")
    return fig


def get_data_from_api(gtin, reg_id):
    url = "http://127.0.0.1:5000/get_data"
    payload = {"gtin": gtin, "reg_id": reg_id}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.warning("Error fetching data from API")
        return None


def get_gtins():
    url = "http://127.0.0.1:5000/get_gtins"
    response = requests.post(url)

    if response.status_code == 200:
        return list(response.json())
    else:
        st.warning("Error fetching data from API")
        return None


def get_reg_ids():
    url = "http://127.0.0.1:5000/get_reg_ids"
    response = requests.post(url)

    if response.status_code == 200:
        reg_ids = list(response.json())
        return reg_ids
    else:
        st.warning("Error fetching data from API")
        return None


st.set_page_config(page_title="MARKING HACK", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Название")

navbar_container = st.container()

with navbar_container:
    col1, col2, col3 = st.columns(3)

    analytics_tab, ranking_tab, tab3 = st.tabs(["Аналитика", "Ранжирование", "Tab 3"])

with analytics_tab:
    reg_ids = st.selectbox("Регион:", get_reg_ids())
    gtins = st.text_input("Товар:", "00C22971781D72C7C475869EC049959A")
    # gtins = st.selectbox("Товар:", "get_gtins()")

    data = get_data_from_api(gtins, int(reg_ids))
    data = data.sort_values('dt')

    if data is not None:
        if len(data) == 0:
            st.warning('Недостаточно данных')
        else:
            st.dataframe(data)
            graph = plot_graph(data, reg_ids, gtins)
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
