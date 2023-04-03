from datetime import datetime as dt
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

print(dt.now(), "Analytics Visited")

st.set_page_config(layout="wide")


def group_region_retails(product_gtin: str, start_date, end_date):
    places_columns = ["Владелец карточки товара", "ИНН ритейлера",
                      "Точка продажи", "Цена", "Количество"]

    # Проданные товары определенного gtin
    sales = closings[(closings['gtin'] == product_gtin) &
                     closings['type_operation'].isin(sale_fields) &
                     (closings["dt"] >= start_date) & (closings["dt"] <= end_date)]

    # Продажи с городами точек продаж
    sales = sales.merge(retails.drop(["inn"], axis=1), left_on='id_sp_',
                        right_on='id_sp_')  # ['id_sp', 'inn', 'region_code']

    # Выделение фич
    grouped = sales.groupby("region_code", group_keys=True)
    # regions = list(groups.keys())

    min_places = []
    max_places = []
    volumes = []
    max_volumes = []
    nans = []
    for name, group in grouped:
        nans.append(np.NaN)

        min_sample = group.sort_values(by="price", ascending=True)[["prid", "inn", "id_sp_", "price", "cnt"]].iloc[:5]
        min_sample.columns = places_columns
        min_places.append(min_sample)

        max_sample = group.sort_values(by="price", ascending=False)[["prid", "inn", "id_sp_", "price", "cnt"]].iloc[:5]
        max_sample.columns = places_columns
        max_places.append(max_sample)

        volume_grouped = group.groupby(by="inn").agg({"cnt": 'sum'})\
            .sort_values(by='cnt', ascending=False).reset_index(names=['ИНН ритейлера'])
        volume_grouped.columns = ["ИНН ритейлера", "Объем продаж"]

        volumes.append(volume_grouped.iloc[0:5])
        max_volumes.append(volume_grouped.iloc[0]["Объем продаж"])

    result = {"regions": list(grouped.groups.keys()),
              "mean_prices": grouped["price"].mean().to_list(),
              "min_prices": grouped["price"].min().to_list(),
              "min_places": min_places,
              "max_prices": grouped["price"].max().to_list(),
              "max_places": max_places,
              "volumes": volumes,
              "max_volumes": max_volumes,
              "nans": nans}

    return result


sale_fields = ['Продажа конечному потребителю в точке продаж',
               'Дистанционная продажа конечному потребителю',
               'Конечная продажа организации', 'Продажи за пределы РФ',
               'Продажа по государственному контракту']

value_type_mapping = {"Минимальная цена": ("min_prices", "min_places"),
                      "Максимальная цена": ("max_prices", "max_places"),
                      "Средняя цена": ("mean_prices", "nans"),
                      "Объем продаж": ("max_volumes", "volumes")}

description = "Государству интересно видеть аномальные показатели продаж для выявления потенциально подозрительных торговых точек.  \n\n" \
              "Для проведения аналитики мы используем показатели цен:  \n" \
              "**Аномально высокую минимальную цену (аномально низкую максимальную)** у ретейлера на товар с конкретным `gtin` относительно окружающих его регионов. " \
              "Смотрим, какой ретейл продал выбранный `gtin` в регионе по минимальной цене в заданный промежуток времени. " \
              "Отображается карта регионов, на которой они имеют соответствующую раскраску, исходя из показателя минимальный цены на данный товар.  \n\n"

st.header("Просмотр показателей ритейлеров")
st.markdown(description, unsafe_allow_html=False)

closings = pd.read_parquet("data/Output_short.parquet", engine="fastparquet")  # "data/Output.parquet"
closings["dt"] = closings["dt"].apply(lambda x: dt.strptime(x, '%Y-%m-%d').date())

products = pd.read_csv("data/Products.csv")
retails = pd.read_csv("data/Places.csv")

okato = pd.read_csv('data/okato.csv')

# product_gtin = st.text_input("GTIN интересующего товара, цены на который будут анализироваться:",
#                              "1248F88441BCFC563FB99D77DB0BB80D")
product_gtin = st.selectbox("GTIN интересующего товара", closings["gtin"].unique())
value_type = st.selectbox("Тип значений для анализа", list(value_type_mapping.keys()))
start_date = st.date_input("Начало периода, в котором рассматриваются продажи", dt(2021, 11, 22))
end_date = st.date_input("Конец периода", dt(2022, 11, 22))

retails_features = group_region_retails(product_gtin, start_date, end_date)

russia_regions = gpd.read_file('data/regions_new.geojson')

dictionary = {"REGION_ID": [], "values": [], "tables": []}

# читаем okato и создаём словарь
regions_mapping = {}
for i in range(len(okato)):
    regions_mapping[okato['ISO'][i]] = okato['ОКАТО'][i]

index = 0
for i in regions_mapping.values():
    if i in retails_features["regions"]:
        index = retails_features["regions"].index(i)

        dictionary["REGION_ID"].append(i)
        dictionary["values"].append(retails_features[value_type_mapping[value_type][0]][index])
        dictionary["tables"].append(retails_features[value_type_mapping[value_type][1]][index])
    else:
        dictionary["REGION_ID"].append(i)
        dictionary["values"].append(np.NaN)
        dictionary["tables"].append(np.NaN)

df = pd.DataFrame(dictionary)

# добавляем уникальные id'шники регионов
# russia_regions['REGION_ID'] = dictionary["REGION_ID"]

russia_regions['REGION_ID'] = russia_regions['ref'].replace(regions_mapping)
russia_regions['REGION_ID'].astype('int64')

m = folium.Map(location=[63.391522, 96.328125], zoom_start=3, tiles="cartodb positron")

rel_ = folium.Choropleth(
    geo_data=russia_regions,
    name='Регионы России',
    data=df,
    columns=['REGION_ID', 'values'],
    key_on='feature.properties.REGION_ID',
    bins=5,
    fill_color='YlOrRd',
    nan_fill_color='white',
    nan_fill_opacity=0.5,
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Регионы России',
    highlight=True,
    show=False
)

okato = okato.set_index('ОКАТО')

marker_cluster = MarkerCluster().add_to(m)

for i in range(len(df)):
    x = okato.loc[int(df.iloc[i]['REGION_ID'])]

    if isinstance(df['tables'].loc[i], pd.DataFrame):
        table = df['tables'].loc[i].to_html()
    else:
        table = None

    if value_type == "Средняя цена":
        popup_desc = f"<font size='+0.5'><strong>Средняя цена: {df['values'].loc[i]}</strong></font>"
    else:
        popup_desc = f"<font size='-5'><strong>{table}</strong></font>"

    if ~np.isnan(df['values'].loc[i]):
        folium.Marker(
            location=[x['Ширина'],
                      x['Долгота']],
            popup=popup_desc,  # что видим, когда нажимаем
            tooltip=f"<font size='+0.5'><strong>{x['Название']}</strong></font>",  # что видим, когда наводим
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(marker_cluster)

rel_.add_to(m)

m.save('maps/analytics_map.html')

st.write(
    "При нажатии на метки регионов отображаются подозрительные продажи в регионе, точки этих продаж, ИНН продающих организаций, производители товара, цены")

components.html(open("maps/analytics_map.html", 'r', encoding='utf-8').read(), height=500)
