# 🧠 XAI Loan Eligibility System

Welcome to the **XAI Loan Eligibility System**, an interactive demo application showcasing the power of **Explainable AI (XAI)** in financial decision-making. This system evaluates loan eligibility using a transparent machine learning pipeline and generates human-readable rationales using a local language model.

---

## 🚀 Project Overview

This project demonstrates how **XAI** techniques can be integrated into real-world applications. Users submit loan applications via a simple web interface, and the system:

- Predicts loan eligibility using a trained ML model
- Explains the decision using **LIME** (Local Interpretable Model-agnostic Explanations)
- Generates a natural-language rationale using a local **Phi-3** model via **Ollama**

---

## 🧰 Tech Stack

| Layer        | Technology Used |
|--------------|-----------------|
| **Frontend** | HTML, CSS, JavaScript (served via FastAPI StaticFiles) |
| **Backend**  | FastAPI, Python 3.11, Uvicorn |
| **ML Model** | scikit-learn (Logistic Regression, ColumnTransformer) |
| **Data**     | pandas, NumPy (synthetic dataset generation) |
| **XAI**      | LIME (local feature attribution) |
| **Database** | SQLite (prediction history) |
| **Validation** | Pydantic v2 |
| **LLM**      | Phi-3 via Ollama (local model for rationale generation) |
| **Infrastructure** | Docker, Docker Compose |
| **Startup**  | Bash (`start.sh` script) |

---

## 🏗️ System Architecture

The system consists of two main services:

1. **FastAPI Backend**
   - Serves the frontend
   - Hosts ML prediction and explanation logic
   - Stores prediction history in SQLite
   - Calls the local LLM for rationale generation

2. **Phi-3 Model via Ollama**
   - Runs as a separate container
   - Provides a local API for generating natural-language explanations

These services are orchestrated using **Docker Compose** and launched via the `start.sh` script.

---

## 📦 Prerequisites

Ensure the following are installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 📁 Directory Structure

| Directory      | Purpose                                  |
|----------------|------------------------------------------|
| `static/`      | Frontend assets (`index.html`)           |
| `data/`        | Application data                         |
| `backups/`     | Backup files                             |
| `ollama_data/` | Model data for Phi-3                     |

---

## 🧪 Required Files

Before running, ensure the following files are present:

- `main.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `static/index.html`

---

## 🚀 Startup Instructions

To launch the system:

```bash
./start.sh

The script will:

Check Docker and Docker Compose availability

Create necessary directories

Verify required files

Stop any running containers

Build and start services

Wait for services to become ready

Display access information

🌐 Access Information
Once started, access the system via:

Service	URL
Web Application	http://localhost:8000
API Documentation	http://localhost:8000/docs
Health Check	http://localhost:8000/health
Phi-3 Model API	http://localhost:11434
