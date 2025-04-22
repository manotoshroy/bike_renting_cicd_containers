FROM python:3.12-slim
LABEL maintainer="manotosh.roy@kwglobal.com"
LABEL version="1.0"
LABEL description="Bike Renting Model API"
# Set the working directory
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .

# Copy the rest of the application code into the container
COPY . .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application will run on
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
