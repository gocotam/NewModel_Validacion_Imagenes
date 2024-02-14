# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
import logging
import traceback
import time
import concurrent.futures
from funcion_endpoints import endpoint_validacion, endpoint_enriquecimiento
from funciones_auxiliares import (autoML_enriquecimiento,
                                  combine_dicts,
                                  normalize_dict,
                                  successful_response,
                                  handle_error,
                                  load_valid_attributes,
                                  autoML_validacion,
                                  medidas_format,
                                  compare_images_with_measurements)
logging.basicConfig(level=logging.INFO)

# Inicialización de la API
app = FastAPI()

# Leemos el json de atributos
atributos_json = load_valid_attributes("atributos_validos.json")

# Funciones auxiliares para aplicar la concurrencia
def generate_one_image(img):
    project, endpoint_id, location = endpoint_validacion()
    predicciones = autoML_validacion(project, endpoint_id, img.URL, location)
    if img.Tipo == "Isometrico":
        response_iso = dict(predicciones[0])
        response = None
    else:
        response = dict(predicciones[0])
        response_iso = None
    return img.ID, img.URL, response, response_iso

def enriquecimiento_one_image(img):
    project, endpoint_id, location = endpoint_enriquecimiento()
    predicciones_enriq = autoML_enriquecimiento(
            project=project,
            endpoint_id=endpoint_id,
            filename=img.URL,
            location=location)
    response = dict(predicciones_enriq[0])
    return response

# Función para la validación de imágenes
def validacion(request:ImageRequest):

    medidas_request = request.Medidas
    medidas_request_format = medidas_format(medidas_request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generate_one_image, img) for img in request.Imagenes]
    
        imagenes = []
        for num, img in enumerate(futures):
            try:
                start = time.time()
                ID, URL, response, response_iso = img.result()
                logging.info(f"Generando imagen {ID}...")
                validaciones_response = {}
                for response_data in [response, response_iso]:
                    if response_data is not None:
                        for name in response_data['displayNames']:
                            validaciones_response[name] = True
                        if response_data == response_iso:
                            validaciones_response["medidas"] = compare_images_with_measurements([medidas_request_format[num]], [URL])[0]
                response_object = {}
                response_object["ID"] = ID
                response_object_status = {}
                if all(value == True for value in validaciones_response.values()):
                    response_object_status["Codigo"] = "Exito"
                else:
                    response_object_status["Codigo"] = "Error"
                response_object["Status"] = response_object_status
                response_object_status["Validaciones"] = validaciones_response
                imagenes.append(response_object)
                end = time.time()
                logging.info(f"Imagen_{num+1}: {end - start}\n")
            except Exception as e:
                logging.error(f"Error: {e}")
    return imagenes

# Función para el enriquecimiento de imágenes
def enriquecimiento(request:ImageRequest):
    global atributos_json

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(enriquecimiento_one_image, img) for img in request.Imagenes]
        
        atributos = []
        list_dict = []
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
                list_dict.append(dict(d.values()))
            except Exception as e:
                logging.error(f"Error: {e}")

        d_agrupados = combine_dicts(*list_dict)
        d_normalizados = normalize_dict(d_agrupados)

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
                    atributo_dict["Status"] = False
                    atributo_dict["Predicciones"] = predicciones
                    atributos.append(atributo_dict)
            else:
                atributo_dict["Atributo"] = atributo
                atributo_dict["Status"] = False
                atributo_dict["Predicciones"] = []
                atributos.append(atributo_dict)
    return atributos

def generate_images(request:ImageRequest):
    return {"Imagenes":validacion(request),
            "Atributos":enriquecimiento(request)}

@app.post("/imgs")
async def enriquecimiento_endpoint(request: ImageRequest=Body(...)):
    try:
        results = generate_images(request)
        response = successful_response(results)
        return response
    except HTTPException as he:
        return handle_error(he)

    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        return handle_error(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    