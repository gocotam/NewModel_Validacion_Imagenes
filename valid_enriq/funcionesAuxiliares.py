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
from google.cloud import vision, storage
import functools
import unicodedata
from classes import ImageRequestValid, ImageRequestEnriq


# Funciones
def validarRequestEnriq(request: ImageRequestEnriq):
    aux = True
    if request.Plantilla.strip() == "" or str(request.Prediccion).strip() == "":
        aux = False
        return aux
    elif len(request.Atributos) == 0:
        aux = False
    else:
        for img in request.Imagenes:
            if any(field.strip() == "" for field in [img.Tipo, img.ID, img.URL, img.URI, img.Base64] if field!=None):
                aux = False
                return aux
    return aux

def validarRequestValid(request:ImageRequestValid):
    aux = True
    if request.Plantilla.strip() == "" or str(request.Prediccion).strip() == "":
        aux = False
        return aux
    elif len(request.Medidas) == 0:
        aux = False 
        return False
    else:
        for img in request.Imagenes:
            if any(field.strip() == "" for field in [img.Tipo, img.ID, img.URL, img.URI, img.Base64] if field!=None):
                aux = False
                return aux
    return aux

def base64ToBytes(base64_str):
    # Decodificar la representación base64 a una cadena de bytes
    imagen_bytes = base64.b64decode(base64_str)
    return imagen_bytes

def obtenerBytesDesdeGcs(uri):
    bucket_name = uri.split("/")[2]
    blob_name = uri.split("/")[3]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    bytes_data = blob.download_as_string()
    return bytes_data

def esUriGcs(string):
    return string.startswith('gs://')

def esBase64(string):
    try:
        base64.b64decode(string)
        return True
    except Exception:
        return False

def esUrl(string):
    return string.startswith("http://") or string.startswith("https://")

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

    if esUrl(filename):
        fileContent = getImageBytesFromUrl(filename)
        encodedContent = base64.b64encode(fileContent).decode("utf-8")
    elif esUriGcs(filename):
        fileContent = obtenerBytesDesdeGcs(filename)
        encodedContent = base64.b64encode(fileContent).decode("utf-8")
    elif esBase64(filename):
        encodedContent = filename

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

    if esUrl(filename):
        fileContent = getImageBytesFromUrl(filename)
        encodedContent = base64.b64encode(fileContent).decode("utf-8")
    elif esUriGcs(filename):
        fileContent = obtenerBytesDesdeGcs(filename)
        encodedContent = base64.b64encode(fileContent).decode("utf-8")
    elif esBase64(filename):
        encodedContent = filename

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

    if esUrl(imageUrl):
        image.source.image_uri = imageUrl
    elif esUriGcs(imageUrl):
        image.source.gcs_image_uri = imageUrl
    elif esBase64(imageUrl):
        image.content = imageUrl

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

def detectLogosUri(uri, forbiddenPhrasesFile):
    """Detects logos in the file located in Google Cloud Storage or on the Web."""
    client = vision.ImageAnnotatorClient()

    forbiddenPhrases = readTextFile(forbiddenPhrasesFile)

    image = vision.Image()

    if esUrl(uri):
        image.source.image_uri = uri
    elif esUriGcs(uri):
        image.source.gcs_image_uri = uri
    elif esBase64(uri):
        image.content = uri

    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    
    success = False 
    detected_logos = []

    for logo in logos:
        logo_text = logo.description.lower()
        if any(phrase.lower() in logo_text for phrase in forbiddenPhrases):
            return False

        success = True
        detected_logos.append(logo_text)
        
    if len(detected_logos) != 1:
        success = False
        
    return success

def findUrls(s):
    regex = r'('
    regex += r'(?:(https?|s?ftp):\/\/)?'
    regex += r'(?:www\.)?'
    regex += r'('
    regex += r'(?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\.)+)'
    regex += r'([A-Z]{2,6})'
    regex += r'|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    regex += r')'
    regex += r'(?::(\d{1,5}))?'
    regex += r'(?:(\/\S+)*)'
    regex += r')'

    find_urls_in_string = re.compile(regex, re.IGNORECASE)
    url = find_urls_in_string.findall(s)
    urls = []
    for u in url:
        urls.append(u[0])
    return urls

def analyzeImageText(imageUrl, forbiddenPhrasesFile, monthsFile):

    forbiddenPhrases = readTextFile(forbiddenPhrasesFile)
    months = readTextFile(monthsFile)

    client = vision.ImageAnnotatorClient()

    image = vision.Image()

    if esUrl(imageUrl):
        image.source.image_uri = imageUrl
    elif esUriGcs(imageUrl):
        image.source.gcs_image_uri = imageUrl
    elif esBase64(imageUrl):
        image.content = imageUrl
        
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        detectedText = texts[0].description 

    forbiddenPhrasesList = []
    for phrase in forbiddenPhrases:
        if phrase.lower() in detectedText.lower():
            forbiddenPhrasesList.append(True)

    # Verificamos si hay palabras probihibas
    forbiddenPhrasesDetected = any(forbiddenPhrasesList)

    # Verificamos si hay referencias a meses
    # datePattern = r"\d{1,2}\s*(?:[./-]\s*\d{1,2}){0,2}" # detecta números solos
    datePattern = r"\d{1,2}\s*[/-]\s*\d{1,2}" # número de uno o dos dígitos, seguido de un separador (barra o guion), y luego otro número de uno o dos dígitos
    monthPattern = r"\b(?:%s)\b" % "|".join(months)
    if re.search(datePattern, detectedText, re.IGNORECASE) or re.search(monthPattern, detectedText, re.IGNORECASE):
        monthDetected = True
    else:
        monthDetected = False

    # Verificamos si hay referencias a porcentajes
    porcentagePattern = r"[0-9,]+[%]"
    porcentageDetected = bool(re.search(porcentagePattern, detectedText))

    # Verificamos si hay referencias a símbolos de monedas
    symbolPattern = r'(US\$|€|¥|£|A\$|C\$|Fr|元|HK\$|NZ\$|\$|₩)'
    symbolDetected = bool(re.search(symbolPattern, detectedText))

    # Verificamos si hay sitios web distintos a liverpool
    urls = findUrls(detectedText)
    urlsDetected = any(url != 'www.liverpool.com.mx' for url in urls)

    return forbiddenPhrasesDetected, monthDetected, porcentageDetected, symbolDetected, urlsDetected
