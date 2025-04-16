import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date
from pyspark.sql.functions import col
from contextlib import contextmanager
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pickle 
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MYSQL_USER = 'root'
MYSQL_PASSWORD = 'root_password'  
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3307
MYSQL_DB = 'covid_db'

MYSQL_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
print(f"MySQL Database URL: {MYSQL_DATABASE_URL}")

engine = create_engine(MYSQL_DATABASE_URL)
print("Engine created")

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'covid19_predictions'

    id = Column(Integer, primary_key=True)
    location = Column(String(255))
    date = Column(Date)
    new_cases_predicted = Column(Float)
    new_deaths_predicted = Column(Float)
    new_vaccinations_predicted = Column(Float)
    new_cases_actual = Column(Float)
    new_deaths_actual = Column(Float)
    new_vaccinations_actual = Column(Float)
    total_cases = Column(Float)
    total_deaths = Column(Float)
    life_expectancy = Column(Float)
    excess_mortality = Column(Float)
    model_version = Column(String(255))  # Ajout de la colonne model_version
    created_at = Column(Date)

    def __repr__(self):
        return f"<Prediction(location='{self.location}', date='{self.date}', new_cases_predicted={self.new_cases_predicted}, new_deaths_predicted={self.new_deaths_predicted}, new_vaccinations_predicted={self.new_vaccinations_predicted})>"

Base.metadata.create_all(engine)
print("Base metadata created")

Session = sessionmaker(bind=engine)
print("Sessionmaker created")

@contextmanager
def session_scope():
    """Fournit une session transactionnelle."""
    session = Session()
    try:
        yield session
        session.commit()
        print("Session committed")
    except Exception:
        session.rollback()
        print("Session rolled back")
        raise
    finally:
        session.close()
        print("Session closed")

def load_data_from_hdfs(spark, hdfs_path):
    """
    Charge les données depuis HDFS.

    Args:
        spark: Session Spark.
        hdfs_path (str): Chemin vers les données dans HDFS.

    Returns:
        DataFrame Spark: Données chargées depuis HDFS.
    """
    try:
        print(f"Loading data from HDFS path: {hdfs_path}")
        df = spark.read.parquet(hdfs_path)
        logger.info(f"Données chargées depuis HDFS : {hdfs_path}")
        print("Data loaded from HDFS successfully")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données depuis HDFS : {e}")
        print(f"Error loading data from HDFS: {e}")
        raise

def preprocess_data(df, location):
    """
    Prétraite les données pour un emplacement spécifique.

    Args:
        df (DataFrame Spark): Données d'entrée.
        location (str): Emplacement pour lequel prétraiter les données.

    Returns:
        DataFrame Pandas: Données prétraitées pour l'emplacement spécifié.
    """
    try:
        print(f"Preprocessing data for location: {location}")
        
        if location.lower() == 'all':
            df_pandas = df.toPandas()
        else:
            df_pandas = df.filter(col("location") == location).toPandas()
        if df_pandas.empty:
            logger.warning(f"Aucune donnée disponible pour l'emplacement : {location}")
            print(f"No data available for location: {location}")
            return None

        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        df_pandas = df_pandas.sort_values('date')
        df_pandas = df_pandas.fillna(0)
        logger.info(f"Données prétraitées pour l'emplacement : {location}")
        print(f"Data preprocessed for location: {location}")
        return df_pandas
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données pour l'emplacement {location}: {e}")
        print(f"Error preprocessing data for location {location}: {e}")
        raise


def train_predict_prophet(data, location, prediction_date, model_version="prophet_v1"):
    """
    Entraîne un modèle Prophet et fait des prédictions pour les nouveaux cas.

    Args:
        data (DataFrame Pandas): Données d'entraînement.
        location (str): Emplacement pour lequel effectuer les prédictions.
        prediction_date (datetime.date): Date for which the prediction is made
        model_version (str): Version du modèle.

    Returns:
        float: Prédiction des nouveaux cas pour le jour suivant.
    """
    try:
        print(f"Training Prophet model for location: {location}, prediction date: {prediction_date}")
        model = Prophet()
        prediction_date = pd.to_datetime(prediction_date)
        train_data = data[data['date'] <= prediction_date]
        model.fit(train_data[['date', 'new_cases']].rename(columns={'date': 'ds', 'new_cases': 'y'}))
        future = pd.DataFrame({'ds': [prediction_date + timedelta(days=1)]})
        forecast = model.predict(future)
        predicted_new_cases = forecast['yhat'].iloc[0]
        if pd.isna(predicted_new_cases):  # Check for NaN
            predicted_new_cases = 0.0
        logger.info(f"Prédiction Prophet pour {location} le {prediction_date + timedelta(days=1)}: {predicted_new_cases:.2f}")
        print(f"Prophet prediction for {location} on {prediction_date + timedelta(days=1)}: {predicted_new_cases:.2f}")
        return predicted_new_cases, model
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement et de la prédiction Prophet pour l'emplacement {location}: {e}")
        print(f"Error training and predicting with Prophet for location {location}: {e}")
        return None, None



def train_predict_xgboost_rf(data, location, prediction_date, model_type='XGBoost', model_version="xgboost_v1"):
    """
    Entraîne un modèle XGBoost ou Random Forest et fait des prédictions pour les nouveaux décès.

    Args:
        data (DataFrame Pandas): Données d'entraînement.
        location (str): Emplacement pour lequel effectuer les prédictions.
        prediction_date (datetime.date): Date for which the prediction is made
        model_type (str): 'XGBoost' ou 'Random Forest'.
        model_version (str): Version du modèle.

    Returns:
        float: Prédiction des nouveaux décès pour le jour suivant.
    """
    try:
        print(f"Training {model_type} model for location: {location}, prediction date: {prediction_date}")
        
        features = ['total_cases', 'new_cases', 'total_deaths', 'life_expectancy', 'excess_mortality', 'new_vaccinations'] # Include new_vaccinations
        target = 'new_deaths' 

      
        if target not in data.columns:
            logger.warning(
                f"La colonne cible 'new_deaths' n'est pas disponible dans les données pour l'emplacement : {location}")
            print(f"Target column 'new_deaths' not available for location: {location}")
            return None

        # Diviser les données en ensembles d'entraînement et de test
        prediction_date = pd.to_datetime(prediction_date)
        train_data = data[data['date'] <= prediction_date]
        X = train_data[features]
        y = train_data[target]

        if model_type == 'XGBoost':
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type doit être 'XGBoost' ou 'Random Forest'")

        model.fit(X, y)

        next_day_data = data[data['date'] == prediction_date + timedelta(days=1)]
        if next_day_data.empty:
            predicted_new_deaths = 0
            print(
                f"No data to predict deaths for {location} on {prediction_date + timedelta(days=1)}.  Setting prediction to 0")
        else:
            next_day_features = next_day_data[features]
            predicted_new_deaths = model.predict(next_day_features)[0]
        if pd.isna(predicted_new_deaths): # Check for NaN
            predicted_new_deaths = 0.0
        logger.info(
            f"Prédiction {model_type} pour {location} le {prediction_date + timedelta(days=1)}: {predicted_new_deaths:.2f}")
        print(f"{model_type} prediction for {location} on {prediction_date + timedelta(days=1)}: {predicted_new_deaths:.2f}")
        return predicted_new_deaths, model
    except Exception as e:
        logger.error(
            f"Erreur lors de l'entraînement et de la prédiction {model_type} pour l'emplacement {location}: {e}")
        print(f"Error training and predicting with {model_type} for location {location}: {e}")
        return None, None


def store_predictions(session, location, prediction_date, new_cases_predicted, new_deaths_predicted,
                      new_vaccinations_predicted, new_cases_actual, new_deaths_actual, new_vaccinations_actual,
                      total_cases, total_deaths, life_expectancy, excess_mortality, model_version):
    try:
        print(f"Storing predictions for location: {location}, date: {prediction_date}")
        prediction = Prediction(
            location=location,
            date=prediction_date,
            new_cases_predicted=new_cases_predicted,
            new_deaths_predicted=new_deaths_predicted,
            new_vaccinations_predicted=new_vaccinations_predicted,
            new_cases_actual=new_cases_actual,
            new_deaths_actual=new_deaths_actual,
            new_vaccinations_actual=new_vaccinations_actual,
            total_cases=total_cases,
            total_deaths=total_deaths,
            life_expectancy=life_expectancy,
            excess_mortality=excess_mortality,
            model_version=model_version
        )
        session.add(prediction)
        session.commit()
        logger.info(f"Prédictions et données stockées dans la base de données pour l'emplacement : {location}, date : {prediction_date}")
        print("Predictions and data stored in database")
    except Exception as e:
        logger.error(f"Erreur lors du stockage des prédictions et des données dans la base de données : {e}")
        print(f"Error storing predictions and data in database: {e}")
        raise

def save_model_to_hdfs(spark, model, model_path, model_name):
    try:
        print(f"Saving model {model_name} to HDFS path: {model_path}")
        
        hadoop_path = spark._jvm.org.apache.hadoop.fs.Path(model_path)
        fs = hadoop_path.getFileSystem(spark._jsc.hadoopConfiguration())
        if not fs.exists(hadoop_path):
            fs.mkdirs(hadoop_path)
            print(f"Created HDFS path: {model_path}")

        model_dir = os.path.join(model_path, model_name)
        model_path_hdfs = spark._jvm.org.apache.hadoop.fs.Path(model_dir)
        fs_output_stream = fs.create(model_path_hdfs, True)
        output_stream = fs_output_stream.getWrappedStream()
        pickle.dump(model, output_stream)
        output_stream.close()
        logger.info(f"Modèle enregistré dans HDFS : {model_dir}")
        print(f"Model saved to HDFS: {model_dir}")

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du modèle dans HDFS : {e}")
        print(f"Error saving model to HDFS: {e}")
        raise



def main(spark, hdfs_path, locations):
    try:
        print("Starting main function")
        df = load_data_from_hdfs(spark, hdfs_path)
        if df is None:
            logger.error("Aucune donnée chargée depuis HDFS. Arrêt du programme.")
            print("No data loaded from HDFS. Exiting program.")
            return

        for location in locations:
            print(f"Processing location: {location}")
            data = preprocess_data(df, location)
            if data is None:
                logger.warning(f"Aucune donnée prétraitée pour l'emplacement : {location}. Passage à l'emplacement suivant.")
                print(f"No preprocessed data for location: {location}. Skipping to next location.")
                continue

            # Iterate through each day in the dataset for a rolling prediction
            for i in range(len(data)):
                prediction_date = data['date'].iloc[i].date()
                new_cases_actual = data['new_cases'].iloc[i]
                new_deaths_actual = data['new_deaths'].iloc[i]
                new_vaccinations_actual = data['new_vaccinations'].iloc[i]
                total_cases = data['total_cases'].iloc[i]
                total_deaths = data['total_deaths'].iloc[i]
                life_expectancy = data['life_expectancy'].iloc[i]
                excess_mortality = data['excess_mortality'].iloc[i]
                location = data['location'].iloc[i]

                new_cases_predicted, prophet_model = train_predict_prophet(data.copy(), location, prediction_date)

                new_deaths_predicted, xgboost_rf_model = train_predict_xgboost_rf(data.copy(), location, prediction_date, model_type='XGBoost')
                new_vaccinations_predicted = 0 
                
            
                with session_scope() as session:
                    store_predictions(session, location, prediction_date, new_cases_predicted, new_deaths_predicted,
                                      new_vaccinations_predicted, new_cases_actual, new_deaths_actual, new_vaccinations_actual,
                                      total_cases, total_deaths, life_expectancy, excess_mortality,
                                      model_version="combined_model_v1")
                                      
                model_base_path = "hdfs://localhost:8020/pandemic/models"  
                if prophet_model:
                    save_model_to_hdfs(spark, prophet_model, model_base_path, f"prophet_{location}_{prediction_date}")
                if xgboost_rf_model:
                    save_model_to_hdfs(spark, xgboost_rf_model, model_base_path, f"xgboost_rf_{location}_{prediction_date}")

        spark.stop()
        print("Spark stopped")
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'exécution du programme principal : {e}")
        print(f"An error occurred during the execution of the main program: {e}")
        raise
if __name__ == "__main__":
    try:
        from pyspark.sql import SparkSession
        from pyspark import SparkConf
        
        conf = SparkConf() 
        conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        conf.set("spark.jars.packages", "mysql:mysql-connector-java:8.0.23")
        conf.set("spark.sql.streaming.checkpointLocation", "hdfs://localhost:8020/checkpoints/mobility")
        conf.set("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020")
        conf.set("spark.driver.host", "localhost")
        conf.set("spark.driver.bindAddress", "0.0.0.0")
        conf.set("spark.ui.port", "4041")
        conf.set("spark.kafka.consumer.request.timeout.ms", "60000")
        conf.set("spark.kafka.consumer.session.timeout.ms", "30000")
        conf.set("spark.sql.streaming.kafka.consumer.poll.ms", "60000")
        spark = SparkSession.builder.appName("Covid19Data")\
        .master("local[*]") \
        .config(conf=conf)\
        .getOrCreate()
        print("Spark session initialized")
        
        hdfs_path = "hdfs://localhost:8020/data/covid19"  

        locations = ['Mexico', 'Canada', 'Morocco']  
        main(spark, hdfs_path, locations)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)