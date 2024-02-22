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
import requests
from PIL import Image
from io import BytesIO
import functools
import unicodedata

# Funciones
def prettyErrors(d:dict) -> str:
    s = "Errores detectados:"
    if all(valor == False for valor in d.values()):
        s += " Ningún error detectado."
    else:
      aux = set([clave for clave, valor in d.items() if valor])
      if aux == {"modelo", "mesa"} or aux == {"modelo"} or aux == {"mesa"}:
        s += " Ningún error detectado."
      else:
        for v in [clave for clave, valor in d.items() if valor]:
            if v in ["modelo", "mesa"]:
                continue
            s += f"\n{v}"
    return s

def stripAccents(s):
   quitandoAcentos =  ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
   return quitandoAcentos.lower()

def stringCamelCase(s):
    camel_case = ''.join(x for x in s.title() if x.isalnum())
    return camel_case[0].lower() + camel_case[1:]

def getImageBytesFromUrl(imageUrl: str) -> bytes:
    with urllib.request.urlopen(imageUrl) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        imageBytes = response.read()
    return imageBytes

def autoMLValidacion(
    project: str,
    endpointId: str,
    filename: str,
    location: str = "us-central1",
    apiEndpoint: str = "us-central1-aiplatform.googleapis.com",
):
    clientOptions = {"api_endpoint": apiEndpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=clientOptions)
    fileContent = getImageBytesFromUrl(filename)

    encodedContent = base64.b64encode(fileContent).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encodedContent,
    ).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.8,
        max_predictions=50,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpointId
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions = response.predictions
    return predictions

def successfulResponseValidacion(results):
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
        "Imagenes": results["Imagenes"]
    }

def successfulResponseEnriquecimiento(results):
    return {
        "Status": {
            "General": "Success",
            "Details": {
                "Atributos": {
                    "Code": "00",
                    "Message": "Generado exitosamente."
                }
            }
        },
        "Atributos": results["Atributos"]
    }

def handleError(error):
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

def autoMLEnriquecimiento(
    project: str,
    endpointId: str,
    filename: str,
    location: str = "us-central1",
    apiEndpoint: str = "us-central1-aiplatform.googleapis.com",
):
    clientOptions = {"api_endpoint": apiEndpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=clientOptions)
    fileContent = getImageBytesFromUrl(filename)

    encodedContent = base64.b64encode(fileContent).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encodedContent,
    ).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=50,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpointId
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions = response.predictions
    return predictions

def combineDicts(*dicts):
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

def normalizeDict(dCombined):
    normalizedDict = {}
    for num, subdict in dCombined.items():
        total = sum(value for values in subdict.values() for value in values)
        normalizedDict[num] = {}
        for key, values in subdict.items():
            normalizedValues = [value / total for value in values if value / total > 0.2]
            if normalizedValues: 
                normalizedDict[num][key] = normalizedValues
    return normalizedDict

@lru_cache(maxsize=3)
def loadValidAttributes(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def medidasFormat(d: dict) -> list:
    medidasRequestFormat = []
    for _, medidas in d.items():
        valores = [medida["valor"] for medida in medidas]
        unidades = [medidas[0]["unidad"]]
        medidasRequestFormat.append({"medida": valores, "unidad": unidades})
    return medidasRequestFormat

def getNumbersAndUnitsFromText(text):
    """
    --> REGEX para buscar números con unidades en el texto, incluyendo decimales <--
    """
    pattern = r'(\b\d+\.\d+\b|\b\d+\b)\s*(cm|mm|in|CM|MM|IN)\b' #centimetros 
    matches = re.findall(pattern, text)
    return matches

def getTextFromImage(imageUrl):
    """
    --> Procesamiento de imagen y detección de texto dentro de la misma <--
    """
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = imageUrl

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        detectedText = texts[0].description 
        numbersAndUnits = getNumbersAndUnitsFromText(detectedText)
        
        return detectedText, numbersAndUnits, texts[1:]

    return None, None, None

def compareMeasurementsWithImage(measurement, num, unit):
    measure = measurement.get("medida", [])
    unitsAllowed = [u.lower() for u in measurement.get("unidad", [])]
    
    unit = unit.lower()

    if float(num) in measure and unit in unitsAllowed:
        return True
    
    return False

def compareImagesWithMeasurements(measurementsList, imageUrls):
    medidasRequestFormat = []
    for _, medidas in measurementsList.items():
        valores = [medida["valor"] for medida in medidas]
        unidades = [medida["unidad"] for medida in medidas]
        if len(set(unidades)) == 1:
            medidasRequestFormat.append({"medida": valores, "unidad": list(set(unidades))})
        else:
            medidasRequestFormat.append({"medida": valores, "unidad": ["Unidades diferentes"]})

    if len(medidasRequestFormat) != len(imageUrls):
        raise ValueError("El número de imágenes no coincide con el número de diccionarios.")
    
    successList = []

    for _, imageUrl in enumerate(imageUrls):
        try:
            detectedText, numbersAndUnits, _ = getTextFromImage(imageUrl)
            if detectedText and numbersAndUnits:
                success = True
                for measurement in medidasRequestFormat:
                    measure = measurement.get("medida", [])
                    for num in measure:
                        numStr = str(num)
                        unitInImage = None
                        for item in numbersAndUnits:
                            if numStr == item[0]:
                                unitInImage = item[1]
                                break
                        if unitInImage is None or not compareMeasurementsWithImage(measurement, numStr, unitInImage):
                            success = False
                            break
                    if success:
                        successList.append(success)
                    else:
                        successList.append(success)
            else:
                successList.append("No se detectó texto o números en la imagen")
        except Exception as e:
            successList.append(f"Error: {str(e)}") 
    return successList

@functools.lru_cache(maxsize=3)
def readTextFile(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def detectLogosUri(uri, forbiddenPhrasesFile, monthsFile):
    """Detects logos in the file located in Google Cloud Storage or on the Web."""
    client = vision.ImageAnnotatorClient()
    
    forbiddenPhrases = readTextFile(forbiddenPhrasesFile)
    allowedWebsites = "www.liverpool.com.mx"
    months = readTextFile(monthsFile)

    response = requests.get(uri)
    imagePil = Image.open(BytesIO(response.content))
    imageBytes = imagePil.convert("RGB").tobytes()
    image = vision.Image(content=imageBytes)

    response = client.text_detection(image=image)
    texts = response.text_annotations[1:]

    forbiddenPhraseDetected = False
    websiteDetected = False
    monthDetected = False

    for text in texts:
        detectedText = text.description

        for phrase in forbiddenPhrases:
            if phrase.lower() in detectedText.lower():
                forbiddenPhraseDetected = True

        if allowedWebsites not in detectedText.lower():
            websiteDetected = True

        datePattern = r"\d{1,2}\s*(?:[./-]\s*\d{1,2}){0,2}"
        monthPattern = r"\b(?:%s)\b" % "|".join(months)
        if re.search(datePattern, detectedText, re.IGNORECASE) or re.search(monthPattern, detectedText, re.IGNORECASE):
            monthDetected = True

    image = vision.Image()
    image.source.image_uri = uri

    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    
    success = len(logos) > 1

    return (forbiddenPhraseDetected, websiteDetected, monthDetected, success)