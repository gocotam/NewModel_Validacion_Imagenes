# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
import logging
import traceback
import time
import concurrent.futures
from funcionEndpoints import endpointValidacion
from funcionesAuxiliares import (successfulResponseValidacion,
                                  handleError,
                                  loadValidAttributes,
                                  autoMLValidacion,
                                  medidasFormat,
                                  compareImagesWithMeasurements,
                                  detectLogosUri)
# Configuración de los logs
logging.basicConfig(level=logging.INFO)

# Inicialización de la API
app = FastAPI()

# Leemos el json de atributos
atributosJson = loadValidAttributes("atributosValidos.json")

# Funciones auxiliares para aplicar la concurrencia
def generateOneImage(img):
    project, endpointId, location = endpointValidacion()
    predicciones = autoMLValidacion(project, endpointId, img.URL, location)
    if img.Tipo == "Isometrico":
        responseIso = dict(predicciones[0])
        response = None
    else:
        response = dict(predicciones[0])
        responseIso = None
    return img.ID, img.URL, response, responseIso

# Función para la validación de imágenes
def validacion(request:ImageRequest):

    medidasRequest = request.Medidas
    medidasRequestFormat = medidasFormat(medidasRequest)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generateOneImage, img) for img in request.Imagenes]
    
        imagenes = []
        for num, img in enumerate(futures):
            try:
                start = time.time()
                ID, URL, response, responseIso = img.result()
                logging.info(f"Generando imagen {ID}...")
                validacionesResponse = {}
                for responseData in [response, responseIso]:
                    if responseData is not None:
                        for name in responseData['displayNames']:
                            validacionesResponse[name] = True
                        if responseData == responseIso:
                            validacionesResponse["medidas"] = compareImagesWithMeasurements([medidasRequestFormat[num]], [URL])[0]
                validacionesResponse["Más de un logo"] = detectLogosUri(URL, "forbiddenPhrases.txt", "months.txt")
                responseObject = {}
                responseObject["ID"] = ID
                responseObjectStatus = {}
                if all(value == True for value in validacionesResponse.values()):
                    responseObjectStatus["Codigo"] = "Exito"
                else:
                    responseObjectStatus["Codigo"] = "Error"
                responseObject["Status"] = responseObjectStatus
                responseObjectStatus["Validaciones"] = validacionesResponse
                imagenes.append(responseObject)
                end = time.time()
                logging.info(f"Imagen_{num+1}: {end - start}\n")
            except Exception as e:
                logging.error(f"Error: {e}")
    return imagenes

def generateImages(request:ImageRequest):
    return {"Imagenes":validacion(request)}

@app.post("/imgs")
async def enriquecimientoEndpoint(request: ImageRequest=Body(...)):
    try:
        results = generateImages(request)
        response = successfulResponseValidacion(results)
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
