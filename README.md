Some good commands to remember :

# Build the Docker image
docker build -t $NAME .

# Run the image 
docker run -p 8888:8888 -v /mnt/c/Users/zakro/Nelson-Dylan-Code/notebooks:/app $NAME

