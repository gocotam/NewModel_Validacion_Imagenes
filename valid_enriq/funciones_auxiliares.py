# Librerías
import urllib.request
import http.client
import typing
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging
import json
from functools import lru_cache
import re
from google.cloud import vision
import math

# Funciones
def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes

def autoML_validacion(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # with open(filename, "rb") as f:
    #     file_content = f.read()
    file_content = get_image_bytes_from_url(filename)

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.7,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    #print("response")
    #print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    return predictions

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
        "Imagenes": results["Imagenes"],
        "Atributos": results["Atributos"]
    }

def handle_error(error):
    code = "01"
    message = "Servicio no disponible."

    if isinstance(error, HTTPException):
        message = str(error.detail)
        code = "01" if error.status_code != 500 else "02"
    else:
        logging.error(f"Error no esperado: {error}")

    return JSONResponse(content={
        "Status": {
            "General": "Error",
            "Details": {
                "Images": {
                    "Code": code,
                    "Message": message
                }
            }
        }
    }, status_code=getattr(error, 'status_code', 500))

def autoML_enriquecimiento(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # with open(filename, "rb") as f:
    #     file_content = f.read()
    file_content = get_image_bytes_from_url(filename)

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.1,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    #print("response")
    #print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    #for prediction in predictions:
    #    print(" prediction:", dict(prediction))
    return predictions

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        for key, value in d.items():
            num, label = key.split(':')
            if num not in combined:
                combined[num] = {}
            if label not in combined[num]:
                combined[num][label] = []
            combined[num][label].append(value)
    return combined

def normalize_dict(d_combined):
    normalized_dict = {}
    for num, subdict in d_combined.items():
        total = sum(value for values in subdict.values() for value in values)
        normalized_dict[num] = {}
        for key, values in subdict.items():
            normalized_values = [value / total for value in values if value / total > 0.2]
            if normalized_values: 
                normalized_dict[num][key] = normalized_values
    return normalized_dict

@lru_cache(maxsize=1)
def load_valid_attributes(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def medidas_format(d: dict) -> list:
    medidas_request_format = []
    for _, medidas in d.items():
        valores = [medida["valor"] for medida in medidas]
        unidades = [medidas[0]["unidad"]]
        medidas_request_format.append({"medida": valores, "unidad": unidades})
    return medidas_request_format

def get_numbers_and_units_from_text(text):
    """
    --> REGEX para buscar números con unidades en el texto, incluyendo decimales <--
    """
    pattern = r'(\b\d+\.\d+\b|\b\d+\b)\s*(cm|mm|in|CM|MM|IN)\b' #centimetros 
    matches = re.findall(pattern, text)
    return matches

def get_text_from_image(image_url):
    """
    --> Procesamiento de imagen y detección de texto dentro de la misma <--
    """
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_url

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        detected_text = texts[0].description 
        numbers_and_units = get_numbers_and_units_from_text(detected_text)
        
        return detected_text, numbers_and_units, texts[1:]

    return None, None, None

def compare_measurements_with_image(measurement, num, unit):
    measure = measurement.get("medida", [])
    units_allowed = [u.lower() for u in measurement.get("unidad", [])]

    if unit in units_allowed:
        for m in measure:
            if math.isclose(float(num), m):
                return True
    return False

def compare_images_with_measurements(measurements_list, image_urls):
    if len(measurements_list) != len(image_urls):
        raise ValueError("El número de imágenes no coincide con el número de diccionarios.")

    success_list = []  

    for idx, image_url in enumerate(image_urls):
        try:
            detected_text, numbers_and_units, _ = get_text_from_image(image_url)
            if detected_text and numbers_and_units:
                success = False  
                for item in numbers_and_units:
                    num, unit = item[0], item[1]
                    unit_in_image = unit.lower()
                    for measurement in measurements_list:
                        if compare_measurements_with_image(measurement, num, unit_in_image):
                            success = True  
                            break
                    if success:
                        break  
                success_list.append(success)  
            else:
                success_list.append(False)  
        except Exception as e:
            success_list.append(False)  
    return success_list  