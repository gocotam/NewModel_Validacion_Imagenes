from pydantic import BaseModel
from typing import Optional

class Medida(BaseModel):
    valor: int
    unidad: str

class Imagen(BaseModel):
    Tipo: str
    ID: str
    URI: Optional[str] = ""
    URL : Optional[str] = ""
    base64_: Optional[str] = ""

#BYTES: Optional[bytes] = b""
class ImageRequestValid(BaseModel):
    Plantilla: str
    Medidas: dict
    Imagenes: list[Imagen]

class ImageRequestEnriq(BaseModel):
    Plantilla: str
    Atributos: dict
    Imagenes: list[Imagen]