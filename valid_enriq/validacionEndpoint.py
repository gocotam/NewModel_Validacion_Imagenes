# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequestValid
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

def string_to_camel_case(string):
    string = ''.join(x for x in string.title() if x.isalnum())
    return string[0].lower() + string[1:]

# Funciones auxiliares para aplicar la concurrencia
def generateOneImage(img):
    project, endpointId, location = endpointValidacion()
    data = img.bytes if img.bytes else img.URI if img.URI.startswith("gs://") else img.URL
    predicciones = autoMLValidacion(project, endpointId, img.URL, location)
    response = dict(predicciones[0])
    # forbiddenPhraseDetected, websiteDetected, monthDetected, logoDetected = detectLogosUri(img.URL, "forbiddenPhrases.txt", "months.txt")

    validacionesResponse = {"fraseProhibida":False, 
                            "paginaWeb":False, 
                            "refAMeses":False, 
                            "masDeUnLogo":False}

    #validacionesResponse = {}
    labels = ["pixelado","corte de extremidades","mesa","mal enfocado","modelo",
              "producto roto","aire","ojos cerrados","etiqueta visible","reflejo","mala iluminacion"]
    
    pretty_errors = "Errores detectados:"
    for name in response["displayNames"]:
        if  name in ["modelo","mesa"]:
            continue
        validacionesResponse[string_to_camel_case(name)] = True
        pretty_errors += f"\n{name}"
    
    if "\n" not in pretty_errors:
        pretty_errors += " Ningún error detectado."
    for label in labels:
        if string_to_camel_case(label) not in validacionesResponse.keys():
            validacionesResponse[string_to_camel_case(label)] = False        
    
    d_aux = {
        "ID": img.ID,
        "URL": img.URL,
        "Tipo": stripAccents(img.Tipo),
        "validacionesResponse": validacionesResponse,
        "DescripcionErrores": pretty_errors,
    }
    return d_aux

# Función para la validación de imágenes
def validacion(request:ImageRequestValid):

    medidasRequest = request.Medidas

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generateOneImage, img) for img in request.Imagenes]
    
        imagenes = []
        i = 1
        for img in futures:#concurrent.futures.as_completed(futures):
            try:
                d_aux = img.result()
                responseObject = {}
                responseObjectStatus = {}
                ID, URL, tipo, validacionesResponse = d_aux.get("ID"), d_aux.get("URL"), d_aux.get("Tipo"), d_aux.get("validacionesResponse")
                errorDescription = d_aux.get("DescripcionErrores")
                logging.info(f"Generando imagen {ID}...")
                if tipo in ["isometrico"]:
                    validacionesResponse["medidas"] = compareImagesWithMeasurements({f"Producto{i}":medidasRequest[f"Producto{i}"]}, [URL])[0]
                    i += 1
                    responseObject["ID"] = ID
                    if any(value == True for value in [elem for elem in validacionesResponse.values() if elem not in ["modelo","mesa"]]):
                        responseObjectStatus["Codigo"] = "Error"
                    else:
                        responseObjectStatus["Codigo"] = "Exito"
                    responseObject["Status"] = responseObjectStatus
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    responseObjectStatus["DescripcionErrores"] = errorDescription
                    imagenes.append(responseObject)
                elif tipo in ["detalle", "principal"]:
                    responseObject["ID"] = ID
                    if any(value == True for value in [elem for elem in validacionesResponse.values() if elem not in ["modelo","mesa"]]):
                        responseObjectStatus["Codigo"] = "Error"
                    else:
                        responseObjectStatus["Codigo"] = "Exito"
                    responseObject["Status"] = responseObjectStatus
                    responseObjectStatus["Validaciones"] = validacionesResponse
                    responseObjectStatus["DescripcionErrores"] = errorDescription
                    imagenes.append(responseObject)
                else:
                    raise HTTPException(status_code=400, detail="Tipo de imagen no válido")
            except Exception as e:
                logging.error(f"Error: {e}")
    return imagenes

def generateImages(request:ImageRequestValid):
    return {"Imagenes":validacion(request)}

@app.post("/imgs")
async def enriquecimientoEndpoint(request: ImageRequestValid=Body(...)):
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
