Some good commands to remember :

# Build the Docker image
docker build -t ${NAME} .

ex: docker build -t jupyter-notebook .

# Run the image 
docker run -p 8888:8888 -v ${PATH_TO_REPO}/notebooks:/app $NAME

ex: docker run -p 8888:8888 -v /mnt/c/Users/zakro/Nelson-Dylan-Code/notebooks:/app jupyter-notebook
