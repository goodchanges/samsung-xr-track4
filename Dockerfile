FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt requirements-temporal.txt ./
ARG ENABLE_TEMPORAL=0
ARG INSTALL_TEST_TOOLS=0
RUN if [ "$ENABLE_TEMPORAL" = "1" ]; then pip install --no-cache-dir -r requirements-temporal.txt; else pip install --no-cache-dir -r requirements.txt; fi
RUN if [ "$INSTALL_TEST_TOOLS" = "1" ]; then pip install --no-cache-dir coverage==7.17.2; fi
COPY code/ code/
COPY tests/ tests/
COPY weights/ weights/
COPY run.sh .
ARG DOWNLOAD_WEIGHTS=0
RUN if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then python code/setup_model.py && if [ "$ENABLE_TEMPORAL" = "1" ]; then python code/setup_temporal.py; fi; fi
RUN chmod +x run.sh
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["./run.sh"]
