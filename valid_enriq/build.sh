# bash build.sh
PROJECT_ID="crp-dev-dig-mlcatalog"
REPO_NAME="validacion-imagenes"
IMAGE_NAME="new-validacion-imagenes-img"
LOCAL_IMAGE_NAME="validacion-imagenes-img:latest"

# Construye la imagen de Docker
docker buildx build --platform linux/amd64 -t $LOCAL_IMAGE_NAME .

IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

# Etiqueta la imagen de Docker
docker tag ${LOCAL_IMAGE_NAME} ${IMAGE_URI}

echo "Subiendo imagen ${LOCAL_IMAGE_NAME} a ${IMAGE_URI}"

# Sube la imagen al registro de contenedores
docker push ${IMAGE_URI}