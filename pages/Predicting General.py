import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import geopandas as gpd
import folium

st.set_page_config(layout="wide")

st.header("Прогнозирование спроса и поставок товаров в стране")
st.markdown(
    "Также мы разработали систему одновременного предсказания ввода товара в оборот и вывода товара из оборота относительно всей страны (без деления по регионам).  "
    "Для этого так же использовались сети LSTM. "
    "Для каждого gtin строится график с предсказаниями ввода и вывода товара на 60 дней вперед. "
    "Для оценки снова была введена метрика. Пусть  " + "\n" +
    r"* $\text{last predicted enty cumsum}$ -- последнее предсказанное значение количества введеного товара. "
    r"* $\text{last predicted sold cumsum}$ -- последнее предсказанное значение количество выведенного из оборота товара  " + "\n" +
    "Тогда мeтрика считается как:  " + "\n" +
    r"$$\text{metric}:=\frac{\text{(last predicted enty cumsum} - \text{last predicted sold cumsum}}{\text{last predicted enty cumsum}}$$" + "\n" +
    "Метрика показывает насколько процентов предсказанный вывод товара из оборота меньше ввода в оборот. "
    "Чем ниже процент, тем больше в стране требуется конкретного товара (отрицательный процент означает, что вывод товара превысит поставки)")


def plot_graph(df):
    curr_data = df[df['type'] == 'real_data'][['cnt_cumsum_entry', 'cnt_cumsum_sold', 'dt']]
    predicted_data = df[df['type'] == 'predicted'][['cnt_cumsum_entry', 'cnt_cumsum_sold', 'dt']]

    test = df[df['type'] == 'test'][['cnt_cumsum_entry', 'cnt_cumsum_sold', 'dt']]

    test['cnt_cumsum_entry'] += curr_data['cnt_cumsum_entry'][-1:].values
    test['cnt_cumsum_sold'] += curr_data['cnt_cumsum_sold'][-1:].values

    fig1 = px.line(curr_data, x='dt', y='cnt_cumsum_sold')
    fig2 = px.line(predicted_data, x='dt', y='cnt_cumsum_sold')
    fig2['data'][0]['line']['color'] = "#ffa500"
    fig2['data'][0]['line']['dash'] = "dot"
    fig5 = px.line(test, x='dt', y='cnt_cumsum_sold')
    fig5['data'][0]['line']['color'] = "#8b00ff"

    fig3 = px.line(curr_data, x='dt', y='cnt_cumsum_entry')
    fig3['data'][0]['line']['color'] = "#ff2b2b"
    fig4 = px.line(predicted_data, x='dt', y='cnt_cumsum_entry')
    fig4['data'][0]['line']['color'] = "#008000"
    fig4['data'][0]['line']['dash'] = "dot"
    fig6 = px.line(test, x='dt', y='cnt_cumsum_entry')
    fig6['data'][0]['line']['color'] = "#9b2d30"

    fig = go.Figure(data=fig1.data + fig2.data + fig3.data + fig4.data + fig5.data + fig6.data)
    fig.update_layout(showlegend=True)
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


data = pd.read_csv("data/global_predicts_test.csv")

# Риск дефицита
st.subheader("Оценка рисков дефицита определенного товара")

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
