import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


class Ranking:
    def __init__(self):
        path_to_closings: str = "Desktop/marking hack/Output.csv"

        self.products = pd.read_csv(path_to_closings)

        # TODO: вынести в конфиг
        self.gtin_column = ["gtin"]
        self.sale_fields = ['Продажа конечному потребителю в точке продаж',
                            'Дистанционная продажа конечному потребителю',
                            'Конечная продажа организации', 'Продажи за пределы РФ',
                            'Продажа по государственному контракту']
        self.fields_mapping = {"prid": 0, "variety": 1, "counts": 2, "prices": 3}

    def extract_prices(self, x):
        # Цены на gtin у каждого производителя
        x = x.drop_duplicates(subset=self.gtin_column)
        x = dict(zip(x["gtin"], x["price"]))

        return x

    def extract_features(self, product_type: str):
        # gtin товаров определенной категории
        product_ids = self.products[self.products['product_short_name'] == product_type]['gtin']

        # Проданные товары определенной категории
        sales = self.products[
            self.products['gtin'].isin(product_ids) & self.products['type_operation'].isin(self.sale_fields)]

        # Поставщики
        prids = list(sales.groupby("prid").groups.keys())

        # Разнообразие gtin у каждого производителя
        variety = sales.groupby("prid", group_keys=False).apply(lambda x: x["gtin"].unique().shape[0])

        # Объемы продаж в единицах товаров
        counts = sales.groupby("prid", group_keys=False)["cnt"].sum()

        # Цены проданных товаров
        prices = sales.groupby("prid", group_keys=False).apply(lambda x: self.extract_prices(x))

        return list(zip(prids, variety, counts, prices))

    def ranking(self, product_type: str = "9199AB529CF62D4BDB7E8B1D7459001D"):
        ranked_list = self.extract_features(product_type)
        for i in tqdm(range(0, len(ranked_list) - 1)):
            for index in range(len(ranked_list) - i - 1):
                # Разнообразие больше следующего
                variety_larger_score = int(
                    ranked_list[index][self.fields_mapping["variety"]] > ranked_list[index + 1][
                        self.fields_mapping["variety"]])

                # Объем продаж больше следующего
                counts_larger_score = int(
                    ranked_list[index][self.fields_mapping["counts"]] > ranked_list[index + 1][
                        self.fields_mapping["counts"]])

                # Сравнение цен на одинаковые товары
                same_products = ranked_list[index][self.fields_mapping["prices"]].keys() & ranked_list[index + 1][
                    self.fields_mapping["prices"]].keys()

                counter = 0
                for product in same_products:
                    # Если цена выше -> поставщик лучше
                    if ranked_list[index][self.fields_mapping["prices"]][product] > \
                            ranked_list[index + 1][self.fields_mapping["prices"]][product]:
                        counter += 1
                # Часть лучших продуктов среди общих
                prices_benefits_score = counter / max(1, len(same_products))

                # Общий скор
                score = variety_larger_score + counts_larger_score + prices_benefits_score

                if score > 1:
                    ranked_list[index], ranked_list[index + 1] = ranked_list[index + 1], ranked_list[index]

        return json.dumps([{"prid": item[0], "variety": item[1], "volume": item[2]} for item in ranked_list])
