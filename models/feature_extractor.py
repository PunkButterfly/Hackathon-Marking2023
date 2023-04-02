import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


class FeatureExtractor:
    def __init__(self):
        # TODO: вынести в конфиг
        path_to_closings: str = "data/Output_short.parquet"  # "data/Output.parquet"
        path_to_products: str = "data/Products.csv"
        path_to_retails: str = "data/Places.csv"

        self.gtin_column = ["gtin"]
        self.sale_fields = ['Продажа конечному потребителю в точке продаж',
                            'Дистанционная продажа конечному потребителю',
                            'Конечная продажа организации', 'Продажи за пределы РФ',
                            'Продажа по государственному контракту']
        self.fields_mapping = {"prid": 0, "variety": 1, "counts": 2, "prices": 3}

        self.closings = pd.read_parquet(path_to_closings, engine="fastparquet")
        self.products = pd.read_csv(path_to_products)
        self.retails = pd.read_csv(path_to_retails)

    def extract_prices(self, x):
        # Цены на gtin у каждого производителя
        x = x.drop_duplicates(subset=self.gtin_column)
        x = dict(zip(x["gtin"], x["price"]))

        return x

    def group_region_retails(self, product_gtin: str):
        # gtin товаров определенной категории
        # product_ids = self.products[self.products['product_short_name'] == product_type]['gtin']

        # Проданные товары определенного gtin
        sales = self.closings[(self.closings['gtin'] == product_gtin) &
                              self.closings['type_operation'].isin(self.sale_fields)]

        # Продажи с городами точек продаж
        sales = sales.merge(self.retails.drop(["inn"], axis=1), left_on='id_sp_', right_on='id_sp_')  # ['id_sp', 'inn', 'region_code']

        # Выделение фич
        grouped = sales.groupby("region_code", group_keys=True)
        # regions = list(groups.keys())

        min_places = []
        max_places = []
        for name, group in grouped:
            min_places.append([sales.iloc[group["price"].idxmin()]["inn"]])
            max_places.append([sales.iloc[group["price"].idxmax()]["inn"]])

        result = {"regions": list(grouped.groups.keys()),
                  "mean_prices": grouped["price"].mean().to_list(),
                  "min_prices": grouped["price"].min().to_list(),
                  "min_places": min_places,
                  "max_prices": grouped["price"].max().to_list(),
                  "max_places": max_places}

        return result

    def ranking(self, product_type: str = "9199AB529CF62D4BDB7E8B1D7459001D"):
        ranked_list = self.extract_features(product_type)

    # def extract_features(self, product_gtin: str):
    # # gtin товаров определенной категории
    # # product_ids = self.products[self.products['product_short_name'] == product_type]['gtin']
    #
    # # Проданные товары определенного gtin
    # sales = self.closings[
    #     (self.closings['gtin'] == product_gtin) & self.closings['type_operation'].isin(self.sale_fields)]
    #
    # # Производители
    # prids = list(sales.groupby("prid").groups.keys())
    #
    # # Разнообразие gtin у каждого производителя
    # variety = sales.groupby("prid", group_keys=False).apply(lambda x: x["gtin"].unique().shape[0])
    #
    # # Объемы продаж в единицах товаров
    # counts = sales.groupby("prid", group_keys=False)["cnt"].sum()
    #
    # # Цены проданных товаров
    # prices = sales.groupby("prid", group_keys=False).apply(lambda x: self.extract_prices(x))
    #
    # return list(zip(prids, variety, counts, prices))

    # def ranking(self, product_type: str = "9199AB529CF62D4BDB7E8B1D7459001D"):
    #     ranked_list = self.extract_features(product_type)
    #     for i in tqdm(range(0, len(ranked_list) - 1)):
    #         for index in range(len(ranked_list) - i - 1):
    #             # Разнообразие больше следующего
    #             variety_larger_score = int(
    #                 ranked_list[index][self.fields_mapping["variety"]] > ranked_list[index + 1][
    #                     self.fields_mapping["variety"]])
    #
    #             # Объем продаж больше следующего
    #             counts_larger_score = int(
    #                 ranked_list[index][self.fields_mapping["counts"]] > ranked_list[index + 1][
    #                     self.fields_mapping["counts"]])
    #
    #             # Сравнение цен на одинаковые товары
    #             same_products = ranked_list[index][self.fields_mapping["prices"]].keys() & ranked_list[index + 1][
    #                 self.fields_mapping["prices"]].keys()
    #
    #             counter = 0
    #             for product in same_products:
    #                 # Если цена выше -> поставщик лучше
    #                 if ranked_list[index][self.fields_mapping["prices"]][product] > \
    #                         ranked_list[index + 1][self.fields_mapping["prices"]][product]:
    #                     counter += 1
    #             # Часть лучших продуктов среди общих
    #             prices_benefits_score = counter / max(1, len(same_products))
    #
    #             # Общий скор
    #             score = variety_larger_score + counts_larger_score + prices_benefits_score
    #
    #             if score > 1:
    #                 ranked_list[index], ranked_list[index + 1] = ranked_list[index + 1], ranked_list[index]
    #
    #     return pd.DataFrame(reversed(ranked_list), columns=['Prid', 'Variety', 'Volume', 'Prices'])
