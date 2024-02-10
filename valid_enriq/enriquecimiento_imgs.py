# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
import logging
import traceback
from funciones_auxiliares import (autoML_enriquecimiento,
                                  combine_dicts,
                                  normalize_dict,
                                  successful_response,
                                  handle_error,
                                  load_valid_attributes)

# Inicialización de la API
app = FastAPI()

# Definimos los parámetros para la predicción
project = "209565165407"
endpoint_id = "9120899752169308160"
location = "us-central1"

# Leemos el json de atributos
atributos_json = load_valid_attributes("atributos_validos.json")

def enriquecimiento(request:ImageRequest):
    global project, endpoint_id, location, atributos_json

    list_dict = []
    for img in request.Imagenes:
        predicciones_enriq = autoML_enriquecimiento(
            project=project,
            endpoint_id=endpoint_id,
            filename=img.URL,
            location=location)
        response = dict(predicciones_enriq[0])
        d = {}
        for name in response['displayNames']:
            if "No Aplica" not in name:
                num = name.split(":")[0]
                confidence = response["confidences"][response["displayNames"].index(name)]
                if num not in d:
                    d[num] = (name, confidence)
                else:
                    if confidence > d[num][1]:
                        d[num] = (name, confidence)
        list_dict.append(dict(d.values()))

    d_agrupados = combine_dicts(*list_dict)
    d_normalizados = normalize_dict(d_agrupados)

    atributos = []
    atributos_request = request.Atributos

    for atributo in atributos_request:
        atributo_dict = {}
        if atributo in atributos_json.values():
            atributo_dict["Atributo"] = atributo
            predicciones = []
            for num, valor_confianzas in d_normalizados.items():
                if atributo == atributos_json[num]:
                    for valor in valor_confianzas:
                        dict_predicciones = {}
                        dict_predicciones["Valor"] = valor
                        dict_predicciones["Confianza"] = sum(valor_confianzas[valor])
                        dict_predicciones["Match"] = valor == atributos_request[atributo]
                        predicciones.append(dict_predicciones)
            list_confianzas = []
            for d in predicciones:
                list_confianzas.append(d["Confianza"])
            if len(list_confianzas) > 0:
                max_confianza = max(list_confianzas)
                for d in predicciones:
                    if d["Confianza"] == max_confianza:
                        if d["Match"]:
                            atributo_dict["Status"] = True
                        else:
                            atributo_dict["Status"] = False
                atributo_dict["Predicciones"] = predicciones
                atributos.append(atributo_dict)
        else:
            atributo_dict["Atributo"] = atributo
            atributo_dict["Status"] = False
            atributo_dict["Predicciones"] = []
            atributos.append(atributo_dict)
    
    return {"Atributos": atributos}

@app.post("/enriq")
async def enriquecimiento_endpoint(request: ImageRequest=Body(...)):
    try:
        results = enriquecimiento(request)
        response = successful_response(results)
        return response
    except HTTPException as he:
        return handle_error(he)

    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        return handle_error(e)

def successful_response(results):
    return {
        "Status": {
            "General": "Success",
            "Details": {
                "Images": {
                    "Code": "00",
                    "Message": "Generado exitosamente."
                }
            }
        },
        "Atributos": results["Atributos"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)