from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Carregar o modelo treinado a partir do arquivo .h5
model = load_model("modelo_seed42.h5")

# Seu scaler (substitua por seu objeto real)
scaler_carregado = joblib.load("meu_scaler_minmax.joblib")


@app.route("/previsao", methods=["GET"])
def prever():
    # Obter os parâmetros da URL
    valores_parametro = request.args.get("valores")

    # Verificar se o parâmetro 'valores' foi fornecido
    if valores_parametro is None:
        return jsonify({"erro": 'O parâmetro "valores" não foi fornecido'}), 400

    # Converter os valores fornecidos em uma lista de floats
    try:
        valores = [float(valor) for valor in valores_parametro.split(",")]
    except ValueError:
        return jsonify({"erro": "Os valores fornecidos não são numéricos"}), 400

    # Normalizar os dados de entrada
    entrada_normalizada = scaler_carregado.transform(np.array(valores).reshape(-1, 1))

    # Converter para um array numpy e adicionar uma dimensão para representar a sequência temporal
    entrada_array = np.array([entrada_normalizada])

    # Fazer a previsão usando o modelo
    previsao = model.predict(entrada_array)

    # Inverter a normalização da previsão
    previsao_desnormalizada = scaler_carregado.inverse_transform(previsao)

    # A previsão desnormalizada é um array, converter para uma lista
    previsao_lista = previsao_desnormalizada.flatten().tolist()

    return jsonify({"entrada": valores, "previsao": previsao_lista})


if __name__ == "__main__":
    app.run(debug=True)
