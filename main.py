import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
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
    ranker.group_region_retails()
    # product_type = st.text_input("Вид товара", value="9199AB529CF62D4BDB7E8B1D7459001D")
    # ranked_data = ranker.ranking(product_type)
    # if ranked_data is not None:
    #     if len(ranked_data) == 0:
    #         st.warning('Недостаточно данных')
    #     else:
    #         st.dataframe(ranked_data)

with tab3:
    # загружаем данные о границах и именах регионов
    russia_regions = gpd.read_file('admin_level_4.geojson')

    # добавляем уникальные id'шники регионов
    russia_regions['REGION_ID'] = np.arange(0, 88)

    # создаём датафрейм со значениями в каждом регионе
    df = pd.DataFrame()
    df['REGION_ID'] = np.arange(0, 88)
    df['value'] = np.linspace(0.0, 1000.0, num=88)

    m = folium.Map(location=[63.391522, 96.328125], zoom_start=3, tiles='cartodb positro')
    # cartodb positro - весит 27 МБ

    rel_ = folium.Choropleth(
        geo_data=russia_regions,
        name='Регионы России',
        data=df,
        columns=['REGION_ID', 'value'],
        key_on='feature.properties.REGION_ID',
        bins=5,
        fill_color='BuGn',
        nan_fill_color='darkblue',
        nan_fill_opacity=0.5,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Регионы России',
        highlight=True,
        show=False
    )

    rel_.add_to(m)

    m.save('map.html')
