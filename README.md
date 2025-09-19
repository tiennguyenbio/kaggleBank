# Bank Deposit Prediction Application

A **Flask web application** to predict whether a client will subscribe to a bank product, containerized with **Docker** for easy deployment.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Technologies](#technologies)  
4. [Project Structure](#project-structure)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [License](#license)  

---

## Overview

Predict client subscription outcomes using a **pre-trained machine learning model**.  
- Manual input for single predictions.  
- JSON file upload for batch predictions.  

---

## Features

- **Single & Batch Predictions**  
- **Data Preprocessing**   
- **Web Interface (Flask)**  
- **Dockerized Deployment**  

---

## Technologies

- Python 3.10+  
- Flask  
- Scikit-learn & Joblib  
- Docker  
- HTML/CSS

---

## Project Structure
```bash
.
├── Dockerfile
├── README.md
├── app.py
├── main.py
├── model/
│   └── pipe.dat.gz
├── ms/
│   ├── init.py
│   ├── routes.py
│   ├── services.py
│   └── templates/
│       ├── index.html
│       └── json_help.html
├── preprocessing.py
├── requirements.txt
├── train.csv
└── EDA.ipynb
```
---

## Run Flask API locally

### 1. Clone the repository
```bash
git clone https://github.com/tiennguyenbio/kaggleBank.git
cd kaggleBank
```

### 2. Create virtual environment
```bash
pip install uv
uv venv myenv --python 3.10     # uv 
source .venv/bin/activate       # Mac/Linux
```
### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```
### 4. Run Flask app
```bash
uv run app.py
```
API will be available at [127.0.0.1:8000](http://127.0.0.1:8000/)

## Run with Docker

### 1. Build Docker image
```bash
docker build -t bank_subscription_app .
```
### 2. Run Docker container
```bash
docker run -d -p 8081:8000 bank_subscription_app
```
API will be available at [127.0.0.1:8081](http://127.0.0.1:8081/)

## Run with Dockerhub

### 1. Pull image from Docker Hub to localhost

```bash
docker pull tiennguyenbio/deposit
```
### 2. Run Docker container
```bash
docker run -p 8081:8000 tiennguyenbio/deposit
```
### 3. Stop, remove, and restart container
```bash
docker ps
```
Get CONTAINER ID

```bash
docker stop container_id
docker rm container_id
```
```bash
docker restart container_id
```