import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.express as px



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

with open("style.css")as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.title("Название")

navbar_container = st.container()

with navbar_container:
    col1, col2, col3 = st.columns(3)

    graph_tab, tab2, tab3 = st.tabs(["Предикты", "Tab 2", "Tab 3"])


with graph_tab:
    st.sidebar.title("Параметры")

    x_axis = st.sidebar.selectbox("Регион:", get_reg_ids())
    y_axis = st.sidebar.selectbox("Товар:", get_gtins())

    data = get_data_from_api(y_axis, int(x_axis))
    data = data.sort_values('dt')
    print(type(data))
    if data is not None:
        if len(data) == 0:
            st.warning('Недостаточно данных')
        else:
            st.dataframe(data)
            graph = plot_graph(data, x_axis, y_axis)
            st.plotly_chart(graph)
with tab2:
    st.header("Tab 2 Content")

with tab3:
    st.header("Tab 3 Content")