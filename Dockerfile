# Use an existing Docker image as a base
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application files into the container at /app
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 8888

# Command to run the application
CMD ["python", "Processing.py"]

