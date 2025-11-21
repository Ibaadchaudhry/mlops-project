FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# Expose port used by Flower
EXPOSE 8080
# default command (serve nothing by default)
CMD ["/bin/bash", "-c", "echo 'Container image ready. Use docker run ...' && sleep infinity"]