# Step 1: Use official lightweight Python image as base OS.
FROM python:3.8

# Step 2. Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Step 3. Install production dependencies.
RUN pip install -r requirements.txt



# Use working directory /app/model
WORKDIR /app/model

# Copy and install required packages
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy all the content of current directory to working directory
COPY . .

# Set env variables for Cloud Run
ENV PORT 80
ENV HOST 0.0.0.0

EXPOSE 80:80

# Step 4: Run the web service on container startup using gunicorn webserver.
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app