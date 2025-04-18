﻿# 📊 COVID-19 Mobility Prediction & Real-Time Dashboard

This project provides a real-time data pipeline and dashboard for analyzing and predicting COVID-19 mobility trends using Kafka, Spark, HDFS, MySQL, Grafana, and Streamlit — all containerized with Docker.

## 🔧 Tech Stack

- **Kafka** – Real-time data ingestion
- **Spark** – Streaming & ML processing
- **HDFS** – Distributed storage
- **MySQL** – Stores cleaned + predicted data
- **Grafana** – Real-time dashboards
- **Streamlit** – Interactive data exploration
- **Docker** – All services containerized

## ⚙️ Architecture

CSV → Kafka → Spark → HDFS ↓ MySQL → Grafana ↓ Streamlit

🌐 Services


Streamlit: http://localhost:8501

Grafana: http://localhost:3000

Spark UI: http://localhost:4040

HDFS UI: http://localhost:9870

MySQL: localhost:3307 (root/root_password)

Kafka: localhost:9092

Made with ❤️ by Mohamed Ettalii – AI & Data Engineering Student
