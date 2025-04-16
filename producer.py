from kafka import KafkaProducer
import requests
import pandas as pd
from io import StringIO

kafka_topic = 'pandemic_covid'
kafka_bootstrap_servers = 'localhost:9092' 

owid_url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'


def fetch_owid_data():
    """
    Récupère les données COVID-19.
    """
    try:
        print("Downloading data...")
        response = requests.get(owid_url)
        response.raise_for_status() 
        print("Data downloaded!")
        return response.text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erreur lors de la récupération des données OWID: {e}")


def produce_kafka_data():
    """
    Produit des messages Kafka avec les données COVID-19.
    """
    producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers,
                             value_serializer=lambda x: x.encode('utf-8'))
    print("Connected to broker")
    data = fetch_owid_data()
    df = pd.read_csv(StringIO(data))
    for _, row in df.iterrows():
        try:
            message = row.to_json()
            producer.send(kafka_topic, value=message)
            print(f"Message sent to Kafka: {message[:50]}...")
        except Exception as e:
            print(f"An error occured while sending the message to Kafka: {e}")
    producer.close()

produce_kafka_data()