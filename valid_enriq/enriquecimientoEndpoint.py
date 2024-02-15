# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
import logging
import traceback
import concurrent.futures
from funcionEndpoints import endpointEnriquecimiento
from funcionesAuxiliares import (autoMLEnriquecimiento,
                                  combineDicts,
                                  normalizeDict,
                                  successfulResponseEnriquecimiento,
                                  handleError,
                                  loadValidAttributes)
# Configuración de los logs
logging.basicConfig(level=logging.INFO)

# Inicialización de la API
app = FastAPI()

# Leemos el json de atributos
atributosJson = loadValidAttributes("atributosValidos.json")

# Funciones auxiliares para aplicar la concurrencia
def enriquecimientoOneImage(img):
    project, endpointId, location = endpointEnriquecimiento()
    prediccionesEnriq = autoMLEnriquecimiento(
            project=project,
            endpointId=endpointId,
            filename=img.URL,
            location=location)
    response = dict(prediccionesEnriq[0])
    return response

# Función para el enriquecimiento de imágenes
def enriquecimiento(request:ImageRequest):
    global atributosJson

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(enriquecimientoOneImage, img) for img in request.Imagenes]
        
        atributos = []
        listDict = []
        for img in concurrent.futures.as_completed(futures):
            try:
                response = img.result()
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
                listDict.append(dict(d.values()))
            except Exception as e:
                logging.error(f"Error: {e}")

        dAgrupados = combineDicts(*listDict)
        dNormalizados = normalizeDict(dAgrupados)

        atributosRequest = request.Atributos

        for atributo in atributosRequest:
            atributoDict = {}
            if atributo in atributosJson.values():
                atributoDict["Atributo"] = atributo
                predicciones = []
                for num, valorConfianzas in dNormalizados.items():
                    if atributo == atributosJson[num]:
                        for valor in valorConfianzas:
                            dictPredicciones = {}
                            dictPredicciones["Valor"] = valor
                            dictPredicciones["Confianza"] = sum(valorConfianzas[valor])
                            dictPredicciones["Match"] = valor == atributosRequest[atributo]
                            predicciones.append(dictPredicciones)
                listConfianzas = []
                for d in predicciones:
                    listConfianzas.append(d["Confianza"])
                if len(listConfianzas) > 0:
                    maxConfianza = max(listConfianzas)
                    for d in predicciones:
                        if d["Confianza"] == maxConfianza:
                            if d["Match"]:
                                atributoDict["Status"] = True
                            else:
                                atributoDict["Status"] = False
                    atributoDict["Predicciones"] = predicciones
                    atributos.append(atributoDict)
                else:
                    atributoDict["Status"] = False
                    atributoDict["Predicciones"] = predicciones
                    atributos.append(atributoDict)
            else:
                atributoDict["Atributo"] = atributo
                atributoDict["Status"] = False
                atributoDict["Predicciones"] = []
                atributos.append(atributoDict)
    return atributos

def generateImages(request:ImageRequest):
    return {"Atributos":enriquecimiento(request)}

@app.post("/imgs")
async def enriquecimientoEndpoint(request: ImageRequest=Body(...)):
    try:
        results = generateImages(request)
        response = successfulResponseEnriquecimiento(results)
        return response
    except HTTPException as he:
        return handleError(he)

    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        return handleError(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
