FROM ubuntu:latest
LABEL authors="aljo9"

ENTRYPOINT ["top", "-b"]

FROM python:3.9

WORKDIR /src

COPY Data_Analysis_Dashboard.ipynb /src/
COPY DashboardBackend.py /src/
COPY requirements.txt /src/
COPY customer_shopping_data.csv /src/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "Data_Analysis_Dashboard.ipnyb"]