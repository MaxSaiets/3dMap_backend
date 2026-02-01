# Використовуємо Python 3.11 (стабільна версія для Open3D)
FROM python:3.11-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Встановлюємо системні залежності для Open3D та Trimesh
# libgl1-mesa-glx потрібен для графічних операцій
# libgomp1 потрібен для OpenMP (паралельні обчислення)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо requirements окремо для кешування шарів Docker
COPY requirements.txt .

# Встановлюємо Python залежності
# --no-cache-dir зменшує розмір образу
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код проекту
COPY . .

# Створюємо користувача (HF Spaces вимагає root-less запуску для безпеки)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces очікує порт 7860
EXPOSE 7860

# Команда запуску
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
