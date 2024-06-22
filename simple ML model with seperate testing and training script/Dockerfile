# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose Prometheus port
EXPOSE 8000

# Run main.py when the container launches
#CMD ["python", "deployed_model.py"]
CMD ["sh", "-c", "python deployed_model.py && tail -f /dev/null"]




