# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequestValid, ImageRequestEnriq
import logging
import traceback
import concurrent.futures
from funcionEndpoints import endpointValidacion, endpointEnriquecimiento
from funcionesAuxiliares import *

# Configuración de los logs
logging.basicConfig(level=logging.INFO)

# Inicialización de la API
app = FastAPI()

# Leemos el json de atributos
atributosJson = loadValidAttributes("atributosValidos.json")

# Funciones auxiliares para aplicar la concurrencia
def generateOneImage(img):
    project, endpointId, location = endpointValidacion()
    info = [k for k in [img.URL, img.URI,img.Base64] if k!=""][0]
    predicciones = autoMLValidacion(project, endpointId, info, location)
    response = dict(predicciones[0])
    logoDetected = detectLogosUri(info, "forbiddenPhrases.txt")
    forbiddenPhraseDetected, monthDetected, porcentageDetected, priceDetected, urlsDetected = analyzeImageText(info, "forbiddenPhrases.txt", "months.txt")

    validacionesResponse = {"fraseProhibida": forbiddenPhraseDetected, 
                            "paginaWeb": urlsDetected, 
                            "refAMeses": monthDetected, 
                            "masDeUnLogo": logoDetected,
                            "porcentajesDetectados": porcentageDetected,
                            "preciosDetectados": priceDetected}
    
    labels = ["pixelado", "corte de extremidades", "mesa", "mal enfocado", "modelo", "producto roto", "aire", "ojos cerrados",
              "etiqueta visible", "reflejo", "mala iluminacion"]

    for name in response["displayNames"]:
        validacionesResponse[stringCamelCase(name)] = True
    
    for label in labels:
        if stringCamelCase(label) not in validacionesResponse.keys():
            validacionesResponse[stringCamelCase(label)] = False
            
    d_aux = {
        "ID": img.ID,
        "Info": info,
        "Tipo": stripAccents(img.Tipo),
        "validacionesResponse": validacionesResponse
    }
    return d_aux
def enriquecimientoOneImage(img):
    project, endpointId, location = endpointEnriquecimiento()
    info = [k for k in [img.URL, img.URI, img.Base64] if k!=""][0]
    prediccionesEnriq = autoMLEnriquecimiento(
            project=project,
            endpointId=endpointId,
            filename=info,
            location=location)
    response = dict(prediccionesEnriq[0])
    return response

# Función para la validación de imágenes
def validacion(request:ImageRequestValid):

    medidasRequest = request.Medidas

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generateOneImage, img) for img in request.Imagenes]
    
        imagenes = []
        i = 1
        for img in futures:
            try:
                d_aux = img.result()
                responseObject = {}
                responseObjectStatus = {}
                ID, info, tipo, validacionesResponse = d_aux.get("ID"), d_aux.get("Info"), d_aux.get("Tipo"), d_aux.get("validacionesResponse")
                logging.info(f"Generando imagen {ID}...")
                if tipo in ["isometrico"]:
                    validacionesResponse["medidas"] = compareImagesWithMeasurements({f"Producto{i}":medidasRequest[f"Producto{i}"]}, [info])[0]
                    i += 1
                    responseObject["ID"] = ID
                    if all(value == True for value in validacionesResponse.values()):
                        responseObjectStatus["Codigo"] = "Exito"
                    else:
                        responseObjectStatus["Codigo"] = "Error"
                    responseObject["Status"] = responseObjectStatus
                    responseObject["DescripcionErrores"] = prettyErrors(validacionesResponse)
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    imagenes.append(responseObject)
                elif tipo in ["detalle", "principal"]:
                    responseObject["ID"] = ID
                    if all(value == True for value in validacionesResponse.values()):
                        responseObjectStatus["Codigo"] = "Exito"
                    else:
                        responseObjectStatus["Codigo"] = "Error"
                    responseObject["Status"] = responseObjectStatus
                    responseObject["DescripcionErrores"] = prettyErrors(validacionesResponse)
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    imagenes.append(responseObject)
                else:
                    raise HTTPException(status_code=400, detail="Tipo de imagen no válido")
            except Exception as e:
                logging.error(f"Error: {e}")
    return imagenes
def generateImagesValid(request:ImageRequestValid):
    return {"Imagenes":validacion(request)}

# Función para el enriquecimiento de imágenes
def enriquecimiento(request:ImageRequestEnriq):
    global atributosJson

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = []
        for img in request.Imagenes:
            if stripAccents(img.Tipo) in ["isometrico", "detalle", "principal"]:
                futures.append(executor.submit(enriquecimientoOneImage, img))
            else:
                raise HTTPException(status_code=400, detail="Tipo de imagen no válido")

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
def generateImagesEnriq(request:ImageRequestEnriq):
    return {"Atributos":enriquecimiento(request)}

# Endpoint para la validación de imágenes
@app.post("/imgs")
async def validacionEndpoint(request: ImageRequestValid=Body(...)):
    if request.Prediccion == True:
        try:
            results = generateImagesValid(request)
            response = successfulResponseValidacion(results)
            return response
        except HTTPException as he:
            return handleError(he)

        except Exception as e:
            logging.error(f"Error: {e}")
            traceback.print_exc()
            return handleError(e)
    else:
        return {
            "Status": {
                "General": "Success",
                "Details": {
                    "Images": {
                        "Code": "00",
                        "Message": "No se solicito predicción."
                    }
                }
            },
            "Imagenes": []
        }

# Endpoint para el enriquecimiento de imágenes
@app.post("/enriq")
async def enriquecimientoEndpoint(request: ImageRequestEnriq=Body(...)):
    if request.Prediccion == True:
        try:
            results = generateImagesEnriq(request)
            response = successfulResponseEnriquecimiento(results)
            return response
        except HTTPException as he:
            return handleError(he)

        except Exception as e:
            logging.error(f"Error: {e}")
            traceback.print_exc()
            return handleError(e)
    else:
        return {
            "Status": {
                "General": "Success",
                "Details": {
                    "Atributos": {
                        "Code": "00",
                        "Message": "No se solicito predicción."
                    }
                }
            },
            "Atributos": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


