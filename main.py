import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import requests

from models.feature_extractor import FeatureExtractor
from utils import plot_graph, get_data, get_gtins, get_reg_ids

data = pd.read_csv("data/week_closed_gtin.csv")
okato = pd.read_csv('data/okato.csv')

extractor = FeatureExtractor()

st.set_page_config(page_title="MARKING HACK", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Название")

navbar_container = st.container()

with navbar_container:
    col1, col2, col3 = st.columns(3)

analytics_tab, ranking_tab, tab3 = st.tabs(["Аналитика", "Рейтинг", "Tab 3"])

with analytics_tab:
    pass
    # reg_ids = st.selectbox("Регион:", get_reg_ids(data))
    # gtins = st.text_input("Товар:", "00C22971781D72C7C475869EC049959A")
    # # gtins = st.selectbox("Товар:", "get_gtins()")
    #
    # data = get_data(data, gtins, int(reg_ids))
    # data = data.sort_values('dt')
    #
    # if data is not None:
    #     if len(data) == 0:
    #         st.warning('Недостаточно данных')
    #     else:
    #         st.dataframe(data)
    #         graph = plot_graph(data)
    #         st.plotly_chart(graph)

with ranking_tab:
    pass
    # extractor.group_region_retails()
    # product_type = st.text_input("Вид товара", value="9199AB529CF62D4BDB7E8B1D7459001D")
    # ranked_data = ranker.ranking(product_type)
    # if ranked_data is not None:
    #     if len(ranked_data) == 0:
    #         st.warning('Недостаточно данных')
    #     else:
    #         st.dataframe(ranked_data)

with tab3:
    product_gtin = st.text_input("GTIN товара:", "1248F88441BCFC563FB99D77DB0BB80D")
    value_type = st.selectbox("Тип значений", ["Минимальная цена", "Максимальная цена", "Средняя цена"])

    value_type_mapping = {"Минимальная цена": "min_prices",
                          "Максимальная цена": "max_prices",
                          "Средняя цена": "mean_prices"}

    retails_features = extractor.group_region_retails(product_gtin)

    russia_regions = gpd.read_file('data/regions_new.geojson')

    dictionary = {"REGION_ID": [], "values": []}

    index = 0
    for i in range(0, 85):
        if i in retails_features["regions"]:
            index = retails_features["regions"].index(i)

            dictionary["REGION_ID"].append(i)
            dictionary["values"].append(retails_features[value_type_mapping[value_type]][index])
        else:
            dictionary["REGION_ID"].append(i)
            dictionary["values"].append(np.NaN)

    df = pd.DataFrame(dictionary)

    # читаем okato и создаём словарь
    regions_mapping = {}
    for i in range(len(okato)):
        regions_mapping[okato['ISO'][i]] = okato['ОКАТО'][i]

    # добавляем уникальные id'шники регионов
    russia_regions['REGION_ID'] = dictionary["REGION_ID"]

    russia_regions['REGION_ID'] = russia_regions['ref'].replace(regions_mapping)
    russia_regions['REGION_ID'].astype('int64')

    m = folium.Map(location=[63.391522, 96.328125], zoom_start=3, tiles="cartodb positron")  # , tiles='cartodb positro'
    # cartodb positro - весит 27 МБ

    rel_ = folium.Choropleth(
        geo_data=russia_regions,
        name='Регионы России',
        data=df,
        columns=['REGION_ID', 'values'],
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

    m.save('maps/map.html')

    components.html(open("maps/map.html", 'r', encoding='utf-8').read())
