# Librerías
from fastapi import FastAPI, Body, HTTPException
from classes import ImageRequest
from funciones_auxiliares import autoML_validacion, successful_response, handle_error
import logging
import traceback
import time
import concurrent.futures

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Inicialización de la API
app = FastAPI()

project = "209565165407"
endpoint_id = "1630331652410441728"
location = "us-central1"

def generate_one_image(img):
    global project, endpoint_id, location
    predicciones = autoML_validacion(project, endpoint_id, img.URL, location)
    response = dict(predicciones[0])
    return img.ID, response

def generate_images(request:ImageRequest):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request.Imagenes)) as executor:
        futures = [executor.submit(generate_one_image, img) for img in request.Imagenes]
    
        imagenes = []
        for num, img in enumerate(concurrent.futures.as_completed(futures)):
            try:
                start = time.time()
                ID, response = img.result()
                logging.info(f"Generando imagen {ID}...")
                validaciones_response = {}
                for name in response['displayNames']:
                    validaciones_response[name] = True 
                validaciones_response["atemporal"] = True
                validaciones_response["sin referencias"] = True
                validaciones_response["sin cruce de marcas"] = True
                validaciones_response["legible"] = True
                validaciones_response["medidas"] = True
                validaciones_response["medidas de la imagen"] = True

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
                print(f"Error: {e}")
    return {"Imagenes":imagenes}

@app.post("/imgs")
async def generate_text_endpoint(request:ImageRequest=Body(...)):
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