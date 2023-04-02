import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size)) # (num_layers * num_directions, batch_size, hidden_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def train_model(train_data_normalized, train_window, epochs = 20):
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    #     if i%25 == 1:
    #         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    return model

def get_predict(model, scaler, train_data_normalized, train_window, fut_pred):
    test_inputs = train_data_normalized[-fut_pred:].tolist()

    model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[-fut_pred:]).reshape(-1, 1)).squeeze()

    return [abs(x) for x in actual_predictions]

def get_data_by_getin(df, gtin):
    return df[df.gtin == gtin].sort_values('dt')

def train_and_predict(curr_data, gtin, col = ''):
    series_data = curr_data[col].values

    test_data_size = int(0.15 * len(series_data))
    train_window = int(0.15 * len(series_data))


    train_data = series_data[:-test_data_size]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = torch.FloatTensor(scaler.fit_transform(train_data.reshape(-1, 1))).view(-1)

    model = train_model(train_data_normalized, train_window)

    predictions_repeats = 60

    predicted_values = np.concatenate(([series_data[-1]],
                                get_predict(model, scaler, train_data_normalized, train_window, fut_pred=predictions_repeats)))
    
    predicted_values[1:] += abs(predicted_values[1] - series_data[-1])
    predicted_values[1:] += abs(predicted_values[1:] - predicted_values[0])

    predicted_dates = curr_data[-1:]['dt'].to_list()

    for i in range(len(predicted_values) - 1):
        predicted_dates.append(predicted_dates[-1] + pd.Timedelta("1 day"))

    predicted_data = pd.DataFrame({col: predicted_values, 'dt': predicted_dates})
    return predicted_data

day_data = pd.read_csv('./rus/day_closed_entity_gtin.csv').drop(columns=['Unnamed: 0'])
day_data['cnt_cumsum_sold'] = day_data['cnt_cumsum_sold'].astype(float)
day_data['cnt_cumsum_entry'] = day_data['cnt_cumsum_entry'].astype(float)
day_data['dt'] = pd.to_datetime(day_data['dt'])

top_100_gtin_sold = day_data.groupby('gtin', as_index=False).cnt_cumsum_sold.sum().sort_values('cnt_cumsum_sold')[-100:].gtin
day_data = day_data[day_data.gtin.isin(top_100_gtin_sold)]

BIG_DATA_FRAME = pd.DataFrame(columns=['gtin', 'cnt_cumsum_sold', 'cnt_cumsum_entry', 'dt', 'type', 'metric'])

err_cnt = 0
cnt = 0

pbar = tqdm(total=100)

for idx, row in tqdm(day_data[['gtin']].drop_duplicates().iterrows()):

    try:

        gtin = row['gtin']

        curr_data = get_data_by_getin(day_data, gtin).sort_values('dt')

        predicted_sold = train_and_predict(curr_data, gtin, 'cnt_cumsum_sold')
        predicted_entry = train_and_predict(curr_data, gtin, 'cnt_cumsum_entry')


        curr_data = curr_data[['gtin', 'cnt_cumsum_sold', 'cnt_cumsum_entry', 'dt']]
        curr_data['type'] = 'real_data'

        predicted_data = predicted_sold.copy()
        predicted_data['cnt_cumsum_entry'] = predicted_entry['cnt_cumsum_entry']
        predicted_data['type'] = 'predicted'
        predicted_data['gtin'] = gtin

        metric = (predicted_data['cnt_cumsum_entry'].iloc[-1] - predicted_data['cnt_cumsum_sold'].iloc[-1])/predicted_data['cnt_cumsum_entry'].iloc[-1]

        curr_data['metric'] = metric
        predicted_data['metric'] = metric

        BIG_DATA_FRAME = pd.concat([BIG_DATA_FRAME, pd.concat([curr_data, predicted_data])], ignore_index=True, copy=False)
        cnt += 1
    except:
        err_cnt += 1
    pbar.update(1)
    pbar.set_description(f"success count {cnt}")

print(f"err_count: {err_cnt}")
print(f"succes count: {cnt}")
BIG_DATA_FRAME.to_csv('big_df_rus.csv', index = False)

# fig1 = px.line(curr_data, x = 'dt', y ='cnt_cumsum_sold')
# fig2 = px.line(predicted_data, x = 'dt', y ='cnt_cumsum_sold')
# fig2['data'][0]['line']['color']="#ffa500"
# fig2['data'][0]['line']['dash']="dot"

# fig3 = px.line(curr_data, x = 'dt', y ='cnt_cumsum_entry')
# fig3['data'][0]['line']['color']="#ff2b2b"
# fig4 = px.line(predicted_data, x = 'dt', y ='cnt_cumsum_entry')
# fig4['data'][0]['line']['color']="#008000"
# fig4['data'][0]['line']['dash']="dot"

# fig = go.Figure(data=fig1.data + fig2.data + fig3.data + fig4.data)
# fig.show()



