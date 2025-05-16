FROM python:3.11.12-slim

WORKDIR /app

#build data input/output folder
RUN mkdir -p data
RUN mkdir -p input_data
RUN mkdir -p output_videos

COPY . /app
#Install sys commands?
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 v4l-utils && apt-get clean
RUN pip install -r ./dependencies/linux-requirements.txt

CMD ["python", "./main.py"]