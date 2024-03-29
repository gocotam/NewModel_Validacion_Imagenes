from pydantic import BaseModel
from typing import Optional

class Medida(BaseModel):
    valor: int
    unidad: str

class Imagen(BaseModel):
    Tipo: str
    ID: str
    URI: Optional[str] = None
    URL : Optional[str] = None
    Base64: Optional[str] = None

class ImageRequestValid(BaseModel):
    Plantilla: str
    Prediccion: bool
    Medidas: dict
    Imagenes: list[Imagen]

class ImageRequestEnriq(BaseModel):
    Plantilla: str
    Prediccion: bool
    Atributos: dict
    Imagenes: list[Imagen]