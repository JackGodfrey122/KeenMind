FROM python:3.7.10-buster

WORKDIR keenmind/

COPY * /keenmind/

RUN apt update -y
RUN apt install libgl1-mesa-glx -y

RUN pip install -r docker-requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
