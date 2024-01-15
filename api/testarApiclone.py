from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import tensorflow as tf
from keras import backend as K
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado a partir do arquivo .h5
model = load_model("modelo_seed42.h5")
scaler_carregado = joblib.load("meu_scaler_minmax.joblib")


@app.route("/previsao", methods=["POST"])
def prever():
    # Receber os dados da requisição em formato JSON
    dados_entrada = request.get_json()

    # Verificar se os dados de entrada estão corretos
    if "valores" not in dados_entrada:
        return jsonify({"erro": "Os valores de entrada não foram fornecidos"}), 400

    # Extrair os valores de entrada
    valores_entrada = dados_entrada["valores"]
    valores_entrada = np.array(valores_entrada).reshape(-1, 1)
    valores_entrada = scaler_carregado.transform(valores_entrada)
    # Verificar se há 3 valores de entrada
    if len(valores_entrada) != 3:
        return jsonify({"erro": "Deve haver exatamente 3 valores de entrada"}), 400

    entrada_array = np.array([valores_entrada])

    previsao = model.predict(entrada_array)

    previsao_desnorm = scaler_carregado.inverse_transform(previsao).tolist()

    # Retornar a previsão em formato JSON
    return jsonify({"previsao": previsao_desnorm})


@app.route("/retreinamento", methods=["POST"])
def retreinar():
    global model
    # Ler o último valor de loss do arquivo
    with open("ultimo_loss.txt", "r") as file:
        ultimo_loss_lido = float(file.read())

    print(f"Último valor de loss lido: {ultimo_loss_lido}")

    dados_json = request.get_json()

    data = np.array(dados_json["valores"]).reshape(-1, 1)
    normalized_data = scaler_carregado.transform(data)
    Data_Power = pd.Series(
        [value for row in normalized_data for value in row],
        name="Daily Power yields (kWh)",
    )
    print(Data_Power)
    X = pd.concat(
        [
            Data_Power.shift(1),
            Data_Power.shift(2),
            Data_Power.shift(3),
        ],
        axis=1,
    )
    y = pd.concat([Data_Power.shift(-3)], axis=1)
    X.dropna(inplace=True)
    y.dropna(subset=["Daily Power yields (kWh)"], inplace=True)
    print(X.shape)
    print(y.shape)
    X = X.to_numpy()
    y = y.to_numpy()
    y = y.flatten()
    print(X)
    print(y)

    # Compilar o modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    # Treinar o modelo com os novos dados
    history = model.fit(X, y, epochs=10, batch_size=32)

    # Salvar o modelo treinado
    if history.history["loss"][-1] < ultimo_loss_lido:
        model.save("seu_modelo_retreinado_API.h5")
        return jsonify(
            {
                "mensagem": "Modelo retreinado com sucesso e o erro caiu em relação ao anterior e será atualizado no BD",
                "valor_Loss_anterior": ultimo_loss_lido,
                "valor_Loss_retreinamento": history.history["loss"][-1],
            }
        )
    else:
        model = load_model("modelo_seed42.h5")
        return jsonify(
            {
                "mensagem": "Modelo retreinado com sucesso, mas o erro não caiu em relação ao anterior. Logo, será mantido o anterior no BD",
                "valor_Loss_anterior": ultimo_loss_lido,
                "valor_Loss_retreinamento": history.history["loss"][-1],
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
