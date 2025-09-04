# Gunakan base image Python yang ringan
FROM python:3.13-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam container
# Ini akan menyalin folder healthcare-patient dan static
COPY . .

# Atur environment variable untuk port
ENV PORT 8501

# Jalankan aplikasi menggunakan streamlit
CMD exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0