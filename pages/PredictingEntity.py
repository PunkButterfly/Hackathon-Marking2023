import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import geopandas as gpd
import folium


def plot_graph(df):
    curr_data = df[df['type'] == 'real_data'][['cnt_cumsum_entry', 'cnt_cumsum_sold', 'dt']]
    predicted_data = df[df['type'] != 'real_data'][['cnt_cumsum_entry', 'cnt_cumsum_sold', 'dt']]

    fig1 = px.line(curr_data, x = 'dt', y ='cnt_cumsum_sold')
    fig2 = px.line(predicted_data, x = 'dt', y ='cnt_cumsum_sold')
    fig2['data'][0]['line']['color']="#ffa500"
    fig2['data'][0]['line']['dash']="dot"

    fig3 = px.line(curr_data, x = 'dt', y ='cnt_cumsum_entry')
    fig3['data'][0]['line']['color']="#ff2b2b"
    fig4 = px.line(predicted_data, x = 'dt', y ='cnt_cumsum_entry')
    fig4['data'][0]['line']['color']="#008000"
    fig4['data'][0]['line']['dash']="dot"

    fig = go.Figure(data=fig1.data + fig2.data + fig3.data + fig4.data)
    return fig


def get_data(data, gtin):
    result = data[data['gtin'] == gtin]
    return result


def get_gtins(data):
    result = list(data['gtin'].unique())
    return result


def get_reg_ids(data):
    result = list(data['reg_id'].unique())
    result = [str(x) for x in result]
    return result


data = pd.read_csv("data/global_predictions.csv")

st.title("Прогнозирование спроса и поставок (cumulative)")

# Риск дефицита
st.subheader("Риск дефицита")

sorted_gtins = data.drop_duplicates(subset=["gtin", "metric"]).sort_values(by="metric", ascending=True)["gtin"]

gtin = st.selectbox("Товар:", sorted_gtins, index=0)
# gtin = st.text_input("Товар:", "8CB88CFCD80739D8171051B5116FA775")

sample = get_data(data, gtin).sort_values('dt')

if sample is not None:
    if len(sample) == 0:
        st.warning('Недостаточно данных')
    else:
        st.text(sample['metric'].values[0])
        graph = plot_graph(sample)
        st.plotly_chart(graph)
