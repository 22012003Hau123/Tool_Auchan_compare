# Sử dụng Python 3.10 slim làm base image
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các system dependencies cần thiết
# poppler-utils: cho pdf2image
# tesseract-ocr: cho pytesseract
# libgl1-mesa-glx & libglib2.0-0: cho opencv
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt vào container
COPY requirements.txt .

# Cài đặt PyTorch CPU only trước để tránh kéo file NVIDIA nặng
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện Python còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào container
COPY . .

# Expose port 8502 cho FastAPI
EXPOSE 8502

# Healthcheck để đảm bảo app đang chạy
HEALTHCHECK CMD curl -sf http://localhost:8502/api/health || exit 1

# Lệnh chạy ứng dụng FastAPI bằng uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8502"]
