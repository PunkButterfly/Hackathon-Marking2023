import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt

print(dt.now(), "Deviance Visited")

st.header("Индекс Хиршмана-Херфиндаля для поиска монополий среди поставщиков")
st.markdown("Мы использовали индекс Хиршмана-Херфиндаля для поиска монополий среди поставщиков.  \n"
            "Для каждого `TNVED10` суммируется общее количество товаров у каждого `INN`.  \n"
            "Далее считается доля `INN`, как доля товаров одного `INN` на сумму товаров.  \n"
            "Пусть у нас есть $I = \overline{1,m}$ разных видов товара (в нашем решении вид товара $\Leftrightarrow$ `TNVED10`).  \n"
            " Пусть так же есть $J = \overline{1,n}$ ИНН, производящих продукцуию типов $I$.  \n"
            "Тогда HHI_I = s_1^2 + s_2^2 + \ldots + s_n^2$ -- индекс Хиршмана-Херфиндаля, где  \n"
            "* $s_j$ - доля товаров фирмы $J$ на общее количетсво количество товаров типа $I$:  \n"
            r"$$s_j = \frac{\text{число товаров типа I у INN = J}}{\sum_j \text{(число товаров типа I)}}$$  " + "\n\n" +
            "С помощью этого подхода мы получаем список `TNVED10`, ранжированный по индексу. После чего можно посмотреть список `INN`, отранжированный по количеству производимой продукции типа `TNVED10`.")


def getGroupHHIInfo(tnved10):
    curGoodsHandbook = goods_handbook[goods_handbook['tnved10'] == tnved10]

    merged_df = pd.merge(goods_data_filtered, curGoodsHandbook, on='gtin')
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


start_date = st.date_input("Начало периода, в котором рассматриваются поставки", dt(2021, 11, 22))
end_date = st.date_input("Конец периода", dt(2022, 11, 22))

goods_handbook = pd.read_csv('data/Products.csv')[['gtin', 'tnved10']]

# goods_data = goods_data[['gtin', 'inn', 'cnt']]
goods_data = pd.read_csv('data/Input.csv')
goods_data["dt"] = goods_data["dt"].apply(lambda x: dt.strptime(x, '%Y-%m-%d').date())

goods_data_filtered = goods_data[(goods_data["dt"] >= start_date) &
                                     (goods_data["dt"] <= end_date)][['dt', 'gtin', 'inn', 'cnt']]

# Предпосчитанное дефолтное значение
try:
    scores = pd.read_csv(f"data/{start_date.strftime('%Y/%m/%d').replace('/', '_')}"
                         f"__{end_date.strftime('%Y/%m/%d').replace('/', '_')}_tnveds_hhi.csv",
                         index_col=0).reset_index(drop=True)
    st.dataframe(scores[scores["HHI"] > 0])
except:
    tnveds = goods_handbook['tnved10'].unique()
    metrics = []

    for item in tnveds:
        metrics.append(getGroupHHIInfo(item)[0])

    scores = pd.DataFrame({"tnveds": tnveds, "HHI": metrics})\
        .sort_values(by=['HHI'], ascending=False).reset_index(drop=True)
    st.dataframe(scores[scores["HHI"] > 0])

#
product_tnved = st.text_input("TNVED товара:", "DB208476C3C890F6A24E6BEAE2348127")

volumes = getGroupHHIInfo(product_tnved)[1]
st.dataframe(volumes)
