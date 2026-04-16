# Usa un'immagine Python ufficiale leggera
FROM python:3.10-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia il file dei requisiti e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice sorgente nel container
COPY . .

# Comando di default: avvia il servizio di monitoring
CMD ["python", "data_drift_monitor.py"]