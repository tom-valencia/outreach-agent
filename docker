# Use Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only, for smaller image)
RUN python -m playwright install --with-deps chromium

# Copy app code
COPY . .

# Run your app (replace with your actual entrypoint)
CMD ["python", "app.py"]
