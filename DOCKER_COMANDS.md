docker build -t my-fastapi-app .
docker run -d -p 8000:8000 --name fastapi-container my-fastapi-app
docker stop fastapi-container
docker rm fastapi-container
