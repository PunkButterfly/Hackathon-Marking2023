import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import geopandas as gpd
import folium

from models.feature_extractor import FeatureExtractor

okato = pd.read_csv('data/okato.csv')
extractor = FeatureExtractor()

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

m.save('maps/analytics_map.html')

components.html(open("maps/analytics_map.html", 'r', encoding='utf-8').read())
