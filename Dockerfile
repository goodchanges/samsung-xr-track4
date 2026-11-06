FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt -c requirements.lock.txt
COPY code/ code/
COPY weights/ weights/
COPY run.sh .
ARG DOWNLOAD_WEIGHTS=0
RUN if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then python code/setup_model.py; fi
RUN chmod +x run.sh
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["./run.sh"]
