# Start with a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your project files into the container
COPY . .

# Expose port 7860 for the FastAPI server (Hugging Face Requirement)
EXPOSE 7860

# Command to run the FastAPI server when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]