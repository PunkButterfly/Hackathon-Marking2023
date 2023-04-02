from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Sample data for demonstration purposes
data = pd.read_csv(r"data\week_closed_gtin.csv")

@app.route('/get_data', methods=['POST'])
def get_data():
    params = request.json
    gtin = params.get('gtin')
    reg_id = params.get('reg_id')
    print('*****************',gtin, reg_id)

    result = data[data['gtin']==gtin]
    result = result[result['reg_id']==reg_id]
    print(result.head())
    return jsonify(result.to_dict())

@app.route('/get_gtins', methods=['POST'])
def get_gtins():
    result = list(data['gtin'].unique())
    return jsonify(result)

@app.route('/get_reg_ids', methods=['POST'])
def get_reg_ids():
    result = list(data['reg_id'].unique())
    result = [str(x) for x in result]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)