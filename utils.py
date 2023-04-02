import pandas as pd




# def get_data_from_api(gtin, reg_id):
#     url = "http://127.0.0.1:5000/get_data"
#     payload = {"gtin": gtin, "reg_id": reg_id}
#     response = requests.post(url, json=payload)
#
#     if response.status_code == 200:
#         return pd.DataFrame(response.json())
#     else:
#         st.warning("Error fetching data from API")
#         return None
#
#
# def get_gtins():
#     url = "http://127.0.0.1:5000/get_gtins"
#     response = requests.post(url)
#
#     if response.status_code == 200:
#         return list(response.json())
#     else:
#         st.warning("Error fetching data from API")
#         return None
#
#
# def get_reg_ids():
#     url = "http://127.0.0.1:5000/get_reg_ids"
#     response = requests.post(url)
#
#     if response.status_code == 200:
#         reg_ids = list(response.json())
#         return reg_ids
#     else:
#         st.warning("Error fetching data from API")
#         return None
