import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
filename = "RFmodel.sav"
model_rf = pickle.load(open(filename,'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

# get data
#####data = request.get_json(force=True)

# convert data into dataframe
####data.update((x, [y]) for x, y in data.items())
####data_df = pd.DataFrame.from_dict(data)
####modelo = data_df[0]
####data_df = data_df.drop(['modelo'], axis=1)

#####print("Modelo:", modelo)

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    print("data_df:", data_df)
    modelo = data_df['modelo']
    
    print("Modelo:", modelo)

    if modelo == '0    titanic':  # flag indica que devemos usar o modelo_rf
        # predictions
        data_df = data_df.drop(['modelo'], axis=1)
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

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

