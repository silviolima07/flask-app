import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np

# load model do dataset Titanic
filename_rf = "RFmodel.sav"
model_rf = pickle.load(open(filename_rf,'rb'))
#
# Modelos para do dataset Imoveis
# Modelo criado sem fazer o StandartScaler
filename_rfr = "RandomForestRegressor.sav"
model_rfr = pickle.load(open(filename_rfr,'rb'))
#
# ['Chácara Santo Antônio', 'Vila Mascote','Campo Belo','Jardim Marajoara','Jardim Paulista','Moema','Vila Romana','Perdizes','Vila_Mariana']

model_Moema = pickle.load(open('Modelo_Bairros/ExtraTreesRegressor-Moema.sav','rb'))

model_Perdizes = pickle.load(open('Modelo_Bairros/RandomForestRegressor-Perdizes.sav','rb'))

model_Jardim_Paulista = pickle.load(open('Modelo_Bairros/ExtraTreesRegressor-Jardim_Paulista.sav','rb'))

model_Vila_Mariana = pickle.load(open('Modelo_Bairros/RandomForestRegressor-Vila_Mariana.sav','rb'))

model_Vila_Mascote = pickle.load(open('Modelo_Bairros/DecisionTreeRegressor-Vila_Mascote.sav','rb'))

model_Jardim_Marajoara = pickle.load(open('Modelo_Bairros/ExtraTreesRegressor-Jardim_Marajoara.sav','rb'))

model_Vila_Romana = pickle.load(open('Modelo_Bairros/ExtraTreesRegressor-Vila_Romana.sav','rb'))

model_Campo_Belo = pickle.load(open('Modelo_Bairros/ExtraTreesRegressor-Campo_Belo.sav','rb'))

model_Chacara_Santo_Antonio = pickle.load(open('Modelo_Bairros/DecisionTreeRegressor-Chácara_Santo_Antônio.sav','rb'))


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
    
    elif ('area_total_clean' in
 colunas):
        #
        #df_modelos = pd.read_csv("modelos_bairros.csv")
        
        lista_bairros = ['Chacara Santo Antonio', 'Vila Mascote','Campo Belo','Jardim Marajoara','Jardim Paulista','Moema','Vila Romana','Perdizes','Vila_Mariana']

        # predictions - regression
        print("Previsao de valor de venda")
        # Salvar a informação do bairro que esta na coluna 0 dos dados enviados - data_df
        bairro = list(data_df.bairro)
        print("Bairro enviado:", bairro)
        # Remover esta coluna dos dados enviado, pois o modelo nao foi treinado com esta informação
        data_df = data_df[['area_total_clean','area_util_clean', 'quarto_clean', 'banheiro_clean', 'vaga_clean']]
        print("Dados enviados: ", data_df)
        #
        if bairro in lista_bairros:
            print("Sim, Moema em bairro")

        print("Model Moema:", model_Moema)
        bairro = str(bairro)
        print ("Bairro:", bairro) # ['Moema']

        if 'Moema' in bairro:  
            reg = model_Moema

        if 'Perdizes' in bairro:
            reg = model_Perdizes

        if 'Brooklin' in bairro:
            reg = model_Brooklin

        if 'Jardim Paulista' in bairro:
            reg = model_Jardim_Paulista

        if 'Pinheiros' in bairro:
            reg = model_Pinheiros

        if 'Vila Mariana' in bairro:
            reg = model_Vila_Mariana

        if 'Vila Nova Conceição' in bairro:
            reg = model_Vila_Nova_Conceicao

        if 'Vila Mascote' in bairro:
            reg = model_Vila_Mascote

        if 'Jardim Marajoara' in bairro:
            reg = model_Jardim_Marajoara 

        if 'Vila Romana' in bairro:
            reg = model_Vila_Romana

        if 'Morumbi' in bairro: 
           reg = model_Morumbi

        if 'Vila Leopoldina' in bairro:
           reg = model_Vila_Leopoldina

        if 'Campo Belo' in bairro:
           reg = model_Campo_Belo

        if 'Santana' in bairro:
           reg = model_Santana

        if 'Chacara Santo Antonio' in bairro:
           reg = model_Chacara_Santo_Antonio
           
        if 'Itaim Bibi' in bairro:
           reg = model_Itaim_Bibi 


        result = reg.predict(data_df)
        #result = model_rfr.predict(data_df)
        print("Result:", result)
        # send back to browser
        #result00 = round(np.expm1(int(result[0])),3)
        #output = {'results' : result00}
        #print("Result após np.exp1m:", result00)
        output = {'results': int(result[0])}  # Original ok
        #output = {'results': float(status)}

        #return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

