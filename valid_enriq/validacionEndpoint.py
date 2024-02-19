# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
import logging
import traceback
import concurrent.futures
from funcionEndpoints import endpointValidacion
from funcionesAuxiliares import (successfulResponseValidacion,
                                  handleError,
                                  loadValidAttributes,
                                  autoMLValidacion,
                                  compareImagesWithMeasurements,
                                  detectLogosUri,
                                  stripAccents)
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
    response = dict(predicciones[0])
    validacionesResponse = {}
    validacionesResponse["Más de un logo"] = detectLogosUri(img.URL, "forbiddenPhrases.txt", "months.txt")
    for name in response["displayNames"]:
        validacionesResponse[name] = True
    d_aux = {
        "ID": img.ID,
        "URL": img.URL,
        "Tipo": stripAccents(img.Tipo),
        "validacionesResponse": validacionesResponse
    }
    return d_aux

# Función para la validación de imágenes
def validacion(request:ImageRequest):

    medidasRequest = request.Medidas

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generateOneImage, img) for img in request.Imagenes]
    
        imagenes = []
        i = 1
        for img in concurrent.futures.as_completed(futures):
            try:
                d_aux = img.result()
                responseObject = {}
                responseObjectStatus = {}
                ID, URL, tipo, validacionesResponse = d_aux.get("ID"), d_aux.get("URL"), d_aux.get("Tipo"), d_aux.get("validacionesResponse")
                logging.info(f"Generando imagen {ID}...")
                if tipo in ["isometrico"]:
                    validacionesResponse["medidas"] = compareImagesWithMeasurements({f"Producto{i}":medidasRequest[f"Producto{i}"]}, [URL])[0]
                    i += 1
                    responseObject["ID"] = ID
                    if all(value == True for value in validacionesResponse.values()):
                        responseObjectStatus["Codigo"] = "Exito"
                    else:
                        responseObjectStatus["Codigo"] = "Error"
                    responseObject["Status"] = responseObjectStatus
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    imagenes.append(responseObject)
                elif tipo in ["detalle", "principal"]:
                    responseObject["ID"] = ID
                    if all(value == True for value in validacionesResponse.values()):
                        responseObjectStatus["Codigo"] = "Exito"
                    else:
                        responseObjectStatus["Codigo"] = "Error"
                    responseObject["Status"] = responseObjectStatus
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    imagenes.append(responseObject)
                else:
                    raise HTTPException(status_code=400, detail="Tipo de imagen no válido")
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
