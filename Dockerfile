# Use Python base image
FROM python:3.8-slim

LABEL VERSION=0.1 \ AUTHOR=MAHESH \ EMAIL=aetherix.in@gmail.com

COPY . /app
# Set the working directory in the container
WORKDIR /app

# Update the repositories and install Java
# RUN apt update -y && apt install awscli -y

# Create and activate virtual environment
RUN python3 -m venv myvenv 
# RUN myvenv\bin\activate 

# Install any needed packages specified in requirements.txt
RUN myvenv\bin\pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Specify the command to run on container start
CMD ["python", "app.py"]
