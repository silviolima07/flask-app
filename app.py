import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
filename_rf = "RFmodel.sav"
model_rf = pickle.load(open(filename_rf,'rb'))
#
filename_et = "ExtraTreesRegressor.sav"
model_et = pickle.load(open(filename_et,'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])


def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    print("Data_df:", data_df) 
    colunas = data_df.columns
    print("Colunas:", colunas)

    if ('Classe' in colunas):
        # predictions - classification 
        result = model_rf.predict(data_df)
    
        # Linhas acrescentadas pois o modelo preve: Sobreviveu ou Morreu
        # E a api espera valores inteiros: 0 e 1
        
        if result[0] == "Sobreviveu":
            status = 1
        else:
            status = 0
        
        # send back to browser
        #output = {'results': int(result[0])}
        output = {'results': int(status)}
    
    elif ('area_total_clean' in colunas):
        # predictions - regression
        print("Previsao de valor de venda")
        print("Dados: ", data_df)
        #   
        result = model_et.predict(data_df)
        print("Result:", result)
        # send back to browser
        output = {'results': int(result[0])}
        #output = {'results': float(status)}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

