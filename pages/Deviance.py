import streamlit as st
import pandas as pd
import numpy as np

goods_data = pd.read_csv('data/Input.csv')
goods_handbook = pd.read_csv('data/Products.csv')

goods_data = goods_data[['gtin', 'inn', 'cnt']]
goods_handbook = goods_handbook[['gtin', 'tnved10']]


def getGroupHHIInfo(tnved10):
    curGoodsHandbook = goods_handbook[goods_handbook['tnved10'] == tnved10]
    merged_df = pd.merge(goods_data, curGoodsHandbook, on='gtin')
    result = merged_df.groupby(['inn'])['cnt'].sum().reset_index()

    HHI = sum((result['cnt'] * 100 / sum(result['cnt'])) ** 2)

    #     I тип — высококонцентрированные рынки: при 1800 < HHI < 10000
    #     II тип — умеренноконцентрированные рынки: при 1000 < HHI < 1800
    #     III тип — низкоконцентрированные рынки: при HHI < 1000

    # if HHI > 5000 and len(result) > 2:
    #     print("Категория товара:", tnved10)
    #     print("Показатель монополизации:", HHI)
    #     print("Число компаний:", len(result))
    #     print(result.sort_values(by=['cnt'], ascending=False))

    return int(HHI), result.sort_values(by=['cnt'], ascending=False)


tnveds = goods_handbook['tnved10'].unique()
metrics = []

for item in tnveds:
    metrics.append(getGroupHHIInfo(item)[0])

df = pd.DataFrame({"tnveds": tnveds, "HHI": metrics}).sort_values(by=['HHI'], ascending=False)

st.dataframe(df[df["HHI"] > 0])

#
product_tnved = st.text_input("TNVED товара:", "DB208476C3C890F6A24E6BEAE2348127")

st.dataframe(getGroupHHIInfo(product_tnved)[1])
