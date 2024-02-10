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

#BYTES: Optional[bytes] = b""
class ImageRequest(BaseModel):
    Plantilla: str
    Medidas: dict
    Atributos: dict
    Imagenes: list[Imagen]