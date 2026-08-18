# 💧 Water Potability MLOps

> An end-to-end **Machine Learning Operations (MLOps)** project for predicting whether water is potable using physicochemical water-quality parameters, while demonstrating **data versioning, reproducible ML pipelines, experiment tracking, model evaluation, and CI testing**.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [MLOps Workflow](#-mlops-workflow)
* [Project Architecture](#-project-architecture)
* [Dataset](#-dataset)
* [Machine Learning Pipeline](#-machine-learning-pipeline)
* [MLOps Components](#-mlops-components)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [DVC Pipeline](#-dvc-pipeline)
* [Experiment Tracking](#-experiment-tracking)
* [MLflow and DagsHub](#-mlflow-and-dagshub)
* [CI Testing](#-ci-testing)
* [Model Evaluation](#-model-evaluation)
* [Installation](#-installation)
* [Running the Project](#-running-the-project)
* [DVC Commands](#-dvc-commands)
* [Experiment Tracking with DVC](#-experiment-tracking-with-dvc)
* [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
* [Reproducibility](#-reproducibility)
* [Key Learnings](#-key-learnings)
* [Future Improvements](#-future-improvements)
* [Author](#-author)

---

## 🔎 Overview

**Water Potability MLOps** demonstrates how a traditional machine-learning project can be transformed into a more reproducible and manageable MLOps workflow.

The project focuses on a binary classification problem:

> **Given several physicochemical properties of a water sample, predict whether the water is potable or not.**

Instead of treating model training as a single notebook-based experiment, this project explores different MLOps practices:

* Dataset version management
* Reproducible ML pipelines
* Data preprocessing automation
* Model training automation
* Model evaluation
* Experiment tracking
* Hyperparameter experimentation
* MLflow tracking
* DagsHub integration
* DVC Live experiment tracking
* Continuous Integration testing

The repository contains multiple experiments demonstrating these concepts independently and a structured DVC pipeline that connects the major ML stages.

---

# 🎯 Problem Statement

Water quality can be evaluated using multiple physicochemical parameters such as:

* pH
* Hardness
* Total Dissolved Solids
* Chloramines
* Sulfate
* Conductivity
* Organic Carbon
* Trihalomethanes
* Turbidity

The goal of this project is to use these measurements to classify a water sample as:

| Potability | Meaning                                           |
| ---------- | ------------------------------------------------- |
| `1`        | Potable / suitable according to the dataset label |
| `0`        | Not potable according to the dataset label        |

The project therefore represents a **binary classification problem**.

> **Note:** This project is an educational machine-learning system and should not be treated as a certified water-safety testing system.

---

# 🎯 Objectives

The main objectives of this project are:

1. Build a machine-learning model for water-potability classification.
2. Separate data collection, preprocessing, training, and evaluation into reproducible stages.
3. Use **DVC** for data and pipeline versioning.
4. Track model experiments using DVC experiments.
5. Track ML experiments using **MLflow**.
6. Explore remote experiment tracking using **DagsHub**.
7. Store model evaluation metrics in a structured format.
8. Demonstrate automated testing using Python unit tests.
9. Make the ML workflow easier to reproduce and maintain.
10. Understand the transition from traditional ML development to MLOps.

---

# 🏗️ MLOps Workflow

```text
                    ┌──────────────────────┐
                    │   Water Dataset      │
                    │ water_potability.csv │
                    └──────────┬───────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Data Collection      │
                  │                         │
                  │ • Load dataset          │
                  │ • Read configuration     │
                  │ • Train/Test Split      │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Data Preprocessing   │
                  │                         │
                  │ • Missing-value check   │
                  │ • Median imputation     │
                  │ • Save processed data   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Model Training       │
                  │                         │
                  │ Random Forest Classifier│
                  │ n_estimators = 300      │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Model Evaluation     │
                  │                         │
                  │ • Accuracy              │
                  │ • Precision             │
                  │ • Recall                │
                  │ • F1 Score              │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Experiment Tracking     │
                  │                         │
                  │ • DVC                   │
                  │ • DVC Live              │
                  │ • MLflow                │
                  │ • DagsHub               │
                  └─────────────────────────┘

                               +
                               │
                               ▼
                  ┌─────────────────────────┐
                  │     CI / Unit Tests     │
                  └─────────────────────────┘
```

---

# 🧩 Project Architecture

```text
                         ┌──────────────────┐
                         │     Dataset      │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Data Collection       │
                    │    data_collection.py    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Raw Train/Test Data    │
                    │        data/raw          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Data Preparation      │
                    │       data_prep.py       │
                    │                          │
                    │ Median Imputation        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Processed Dataset      │
                    │     data/processed       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Model Building       │
                    │    model_building.py     │
                    │                          │
                    │ Random Forest Classifier │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       model.pkl          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Model Evaluation     │
                    │      model_eval.py       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       metrics.json       │
                    │                          │
                    │ Accuracy                 │
                    │ Precision                │
                    │ Recall                   │
                    │ F1 Score                 │
                    └──────────────────────────┘
```

---

# 📊 Dataset

The project uses a water-potability dataset containing water-quality measurements and a binary `Potability` target.

### Features

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| `ph`              | Acidity/alkalinity level of water |
| `Hardness`        | Calcium and magnesium hardness    |
| `Solids`          | Total dissolved solids            |
| `Chloramines`     | Chloramine concentration          |
| `Sulfate`         | Sulfate concentration             |
| `Conductivity`    | Electrical conductivity           |
| `Organic_carbon`  | Organic carbon concentration      |
| `Trihalomethanes` | Trihalomethane concentration      |
| `Turbidity`       | Water turbidity                   |
| `Potability`      | Target variable                   |

### Target

```text
Potability = 0 → Non-potable
Potability = 1 → Potable
```

---

# 🤖 Machine Learning Pipeline

The main pipeline consists of four stages.

## 1. Data Collection

The data collection stage:

* Loads the CSV dataset.
* Reads the test-size configuration from `params.yaml`.
* Splits the dataset into training and testing sets.
* Uses a fixed `random_state=42`.
* Stores the resulting datasets inside `data/raw`.

The configured test size is:

```yaml
data_collection:
  test_size: 0.25
```

This results in a **75/25 train-test split**.

---

## 2. Data Preprocessing

The preprocessing stage loads the raw training and testing datasets.

Missing values are handled using **median imputation**.

Conceptually:

```text
Missing Value
      │
      ▼
Calculate Column Median
      │
      ▼
Replace Missing Value
      │
      ▼
Processed Dataset
```

The processed files are stored in:

```text
data/processed/
```

---

## 3. Model Training

The project uses:

```text
RandomForestClassifier
```

The number of trees is configurable through:

```yaml
model_building:
  n_estimators: 300
```

The model uses `Potability` as the target column.

```text
Features
   │
   ├── pH
   ├── Hardness
   ├── Solids
   ├── Chloramines
   ├── Sulfate
   ├── Conductivity
   ├── Organic Carbon
   ├── Trihalomethanes
   └── Turbidity
             │
             ▼
      Random Forest
             │
             ▼
       Potability
```

The trained model is serialized using Python `pickle`.

---

## 4. Model Evaluation

The trained model is evaluated on the processed test dataset.

The following metrics are calculated:

### Accuracy

Measures the overall proportion of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted-positive samples are actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual-positive samples were successfully detected.

```text
Recall = TP / (TP + FN)
```

### F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

The metrics are stored in:

```text
metrics.json
```

---

# ⚙️ MLOps Components

This repository demonstrates multiple MLOps technologies.

| Component                  | Technology      | Purpose                             |
| -------------------------- | --------------- | ----------------------------------- |
| Data Versioning            | DVC             | Track datasets and pipeline outputs |
| Pipeline Management        | DVC             | Reproduce ML stages                 |
| Experiment Tracking        | DVC Experiments | Compare model experiments           |
| Experiment Metrics         | DVC Live        | Track metrics across experiments    |
| Experiment Tracking        | MLflow          | Track ML experiments                |
| Remote Experiment Tracking | DagsHub         | Collaborate on MLflow experiments   |
| Model Serialization        | Pickle          | Save trained model                  |
| Data Processing            | Pandas          | Data manipulation                   |
| Machine Learning           | Scikit-learn    | Model training                      |
| Configuration              | YAML            | Store experiment parameters         |
| Testing                    | PyTest          | Unit testing                        |
| Code Quality               | Flake8          | Linting                             |
| Automation                 | Makefile        | Common development commands         |

---

# 📁 Project Structure

```text
water_potability_MLOPS/
│
├── CI_Demo/
│   ├── example.py
│   ├── test_unit.py
│   ├── steps.txt
│   └── screenshots/
│
├── DVC project1/
│   ├── data/
│   │
│   ├── src/
│   │   ├── data_collection.py
│   │   ├── data_prep.py
│   │   ├── model_building.py
│   │   └── model_eval.py
│   │
│   ├── dvc.yaml
│   ├── dvc.lock
│   ├── params.yaml
│   ├── metrics.json
│   ├── model.pkl
│   └── water_potability.csv
│
├── Experiment_using_Dragshub/
│   ├── mlflow_exp_dagshub/
│   ├── ml_flow2/
│   ├── Readme.txt
│   └── screenshots/
│
├── Exp_autologging_using_mlflow/
│   ├── mlruns/
│   └── Readme.txt
│
├── experiment-tracking-using-DVC/
│   ├── data/
│   ├── dvclive/
│   ├── src/
│   ├── Readme
│   └── dvc.yaml
│
├── mlruns/
│
└── water-potability-version-management/
    ├── data/
    ├── docs/
    ├── dvc.yaml
    ├── dvc.lock
    ├── Makefile
    └── LICENSE
```

---

# 🔄 DVC Pipeline

The structured DVC pipeline contains four stages:

```text
data_collection
       │
       ▼
pre_processing
       │
       ▼
model_building
       │
       ▼
model_eval
```

The pipeline is defined using:

```text
dvc.yaml
```

### Stage 1 — Data Collection

```text
src/data_collection.py
        ↓
data/raw/
```

### Stage 2 — Preprocessing

```text
src/data_prep.py
        ↓
data/processed/
```

### Stage 3 — Model Building

```text
src/model_building.py
        ↓
model.pkl
```

### Stage 4 — Model Evaluation

```text
src/model_eval.py
        ↓
metrics.json
```

This makes the ML workflow reproducible instead of requiring every stage to be manually executed.

---

# 📌 DVC Configuration

The main parameters are stored in:

```text
params.yaml
```

Current configuration:

```yaml
data_collection:
  test_size: 0.25

model_building:
  n_estimators: 300
```

This separates experiment configuration from the source code.

For example, the Random Forest can be changed from:

```yaml
n_estimators: 300
```

to:

```yaml
n_estimators: 500
```

without modifying the training code.

---

# 🧪 Experiment Tracking with DVC

The project also demonstrates DVC experiment tracking.

Different values of `n_estimators` were experimented with and compared using metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

Example experiment results recorded in the repository include:

| Experiment   | n_estimators | Accuracy | Precision |  Recall |      F1 |
| ------------ | -----------: | -------: | --------: | ------: | ------: |
| Experiment 1 |          100 |  0.67378 |   0.60714 | 0.34836 | 0.44271 |
| Experiment 2 |          500 |  0.68902 |   0.65152 | 0.35246 | 0.45745 |
| Experiment 3 |          200 |  0.67530 |   0.61151 | 0.34836 | 0.44386 |

These experiments demonstrate how model hyperparameters can be changed and compared systematically rather than manually recording results.

---

# 📈 DVC Live

The repository contains an experiment-tracking setup using **DVC Live**.

DVC Live can be used to record metrics from ML training runs and visualize/compare experiments.

The project records metrics including:

```text
accuracy
precision
recall
f1 score
```

This makes it easier to understand how changing model parameters affects performance.

---

# 🧬 MLflow Experiment Tracking

The project also explores **MLflow** for experiment tracking.

MLflow is used to maintain information about machine-learning experiments such as:

* Parameters
* Metrics
* Model-related information
* Experiment runs

The repository contains an `mlruns` directory demonstrating MLflow's local tracking structure.

An additional experiment demonstrates MLflow autologging.

---

# 🌐 MLflow + DagsHub

The project also demonstrates integration between:

```text
MLflow
   +
DagsHub
```

The workflow explored in the repository is:

```text
Train Model
     │
     ▼
MLflow Experiment
     │
     ▼
DagsHub
     │
     ▼
Remote Experiment Tracking
```

The DagsHub experiment demonstrates how MLflow experiments can be connected to a remote collaborative environment.

This is particularly useful when multiple people need access to the same experiment history.

---

# 🧪 CI Testing

The repository includes a dedicated:

```text
CI_Demo/
```

directory.

It contains Python unit-test examples and supporting screenshots demonstrating a Continuous Integration workflow.

Main files include:

```text
example.py
test_unit.py
steps.txt
```

The purpose is to demonstrate how automated tests can be incorporated into an ML development workflow.

A typical MLOps workflow can therefore become:

```text
Developer Push
      │
      ▼
Run Tests
      │
      ▼
Validate Code
      │
      ▼
Build / Train
      │
      ▼
Evaluate Model
```

---

# 🧰 Makefile

The project also contains a `Makefile` in the version-management project.

It provides commands for common development operations such as:

* Installing requirements
* Creating environments
* Generating data
* Cleaning compiled Python files
* Running linting
* Synchronizing data with Amazon S3

Example commands:

```bash
make requirements
```

Install project dependencies.

```bash
make data
```

Generate/process dataset artifacts.

```bash
make lint
```

Run Flake8 linting.

```bash
make clean
```

Remove compiled Python artifacts.

The Makefile also contains optional S3 synchronization commands.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Data Processing

* Pandas
* NumPy

### MLOps

* DVC
* DVC Live
* MLflow
* DagsHub

### Configuration

* YAML

### Testing

* PyTest

### Code Quality

* Flake8

### Version Control

* Git
* GitHub

### Optional Cloud Storage

* Amazon S3

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd water_potability_MLOPS
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required packages used by the project:

```bash
pip install pandas numpy scikit-learn pyyaml dvc mlflow pytest flake8 dvclive
```

If the project contains a `requirements.txt`, it can instead be installed using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the DVC Pipeline

Navigate to the DVC project:

```bash
cd "DVC project1"
```

Initialize DVC if required:

```bash
dvc init
```

Then reproduce the pipeline:

```bash
dvc repro
```

DVC will execute the stages according to their dependencies.

```text
Data Collection
       ↓
Preprocessing
       ↓
Model Training
       ↓
Model Evaluation
```

---

# 🔬 Running Individual Stages

## Data Collection

```bash
python src/data_collection.py
```

Creates:

```text
data/raw/
├── train.csv
└── test.csv
```

---

## Data Preprocessing

```bash
python src/data_prep.py
```

Creates:

```text
data/processed/
├── train_processed.csv
└── test_processed.csv
```

---

## Model Training

```bash
python src/model_building.py
```

Creates:

```text
model.pkl
```

---

## Model Evaluation

```bash
python src/model_eval.py
```

Creates:

```text
metrics.json
```

---

# 🔁 Reproducing the Complete Workflow

The recommended approach is:

```bash
dvc repro
```

DVC uses the pipeline definition and dependencies to determine which stages need to be executed.

This provides a reproducible workflow:

```text
Dataset
   ↓
DVC
   ↓
Data Collection
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Metrics
```

---

# 📊 Viewing Metrics

After model evaluation, the metrics are stored in:

```text
metrics.json
```

Example structure:

```json
{
    "acc": 0.68,
    "pre": 0.65,
    "recall": 0.35,
    "f1_score": 0.45
}
```

The exact values depend on the dataset and experiment configuration.

---

# 🧪 DVC Experiment Commands

Create a new experiment:

```bash
dvc exp run
```

View experiments:

```bash
dvc exp show
```

Remove an experiment:

```bash
dvc exp remove <experiment-name>
```

Apply an experiment:

```bash
dvc exp apply <experiment-name>
```

This allows different hyperparameter configurations to be tested without permanently changing the main workspace.

---

# ⚙️ Hyperparameter Experimentation

The primary configurable model parameter is:

```yaml
model_building:
  n_estimators: 300
```

For example:

```yaml
model_building:
  n_estimators: 100
```

or:

```yaml
model_building:
  n_estimators: 500
```

Then execute:

```bash
dvc exp run
```

and compare:

```bash
dvc exp show
```

This creates a systematic workflow for model experimentation.

---

# 🧠 Why MLOps?

A traditional ML project often looks like:

```text
Notebook
   ↓
Train Model
   ↓
Save Model
   ↓
Done
```

This approach becomes difficult when:

* Dataset changes
* Model parameters change
* Multiple experiments are performed
* Multiple developers collaborate
* Results need to be reproduced
* Models need to be compared
* Data needs version control

An MLOps workflow addresses these problems:

```text
Data Versioning
       +
Pipeline Automation
       +
Experiment Tracking
       +
Model Evaluation
       +
Testing
       +
Reproducibility
```

---

# 🔄 Traditional ML vs MLOps

| Traditional ML                    | This Project      |
| --------------------------------- | ----------------- |
| Manual data handling              | DVC               |
| Manual pipeline execution         | DVC Pipeline      |
| Manual experiment records         | DVC Experiments   |
| Manual metric recording           | DVC Live / MLflow |
| Dataset difficult to version      | DVC               |
| Configuration mixed with code     | `params.yaml`     |
| Manual testing                    | Unit Tests        |
| Local experiment tracking         | MLflow            |
| Collaborative experiment tracking | DagsHub           |

---

# 🔐 Reproducibility

One of the primary goals of this project is reproducibility.

The pipeline separates:

```text
Code
Parameters
Data
Model
Metrics
```

For example:

```text
params.yaml
      │
      ▼
DVC Pipeline
      │
      ├── Data Collection
      ├── Preprocessing
      ├── Training
      └── Evaluation
             │
             ▼
         metrics.json
```

A change in a parameter can therefore trigger the relevant pipeline stages and generate updated metrics.

---

# 📌 Key Features

* ✅ Water potability classification
* ✅ Random Forest machine-learning model
* ✅ Automated data splitting
* ✅ Missing-value handling
* ✅ Configurable model parameters
* ✅ DVC pipeline
* ✅ DVC data/version management
* ✅ DVC experiment tracking
* ✅ DVC Live metrics
* ✅ MLflow experiment tracking
* ✅ MLflow autologging experiment
* ✅ DagsHub integration
* ✅ Model evaluation
* ✅ JSON-based metric storage
* ✅ Unit-testing demonstration
* ✅ Flake8 linting
* ✅ Makefile automation
* ✅ Optional S3 data synchronization
* ✅ Reproducible ML workflow

---

# 📚 What This Project Demonstrates

This project goes beyond simply training a machine-learning model.

It demonstrates the progression:

```text
Machine Learning
       │
       ▼
Data Processing
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Experiment Tracking
       │
       ▼
Data Versioning
       │
       ▼
Pipeline Automation
       │
       ▼
Testing / CI
       │
       ▼
MLOps
```

---

# 🚧 Future Improvements

The project can be extended into a more production-oriented MLOps system by adding:

### 1. CI/CD Pipeline

Automate:

```text
Git Push
   ↓
Unit Tests
   ↓
Linting
   ↓
DVC Pipeline
   ↓
Model Evaluation
   ↓
Deployment
```

### 2. Model Serving

Deploy the trained model using:

* FastAPI
* Flask
* Streamlit

### 3. Containerization

Add:

```text
Dockerfile
docker-compose.yml
```

to package the application.

### 4. Model Registry

Use MLflow Model Registry to manage:

```text
Development
     ↓
Staging
     ↓
Production
```

### 5. Model Monitoring

Add monitoring for:

* Prediction distribution
* Data drift
* Feature drift
* Model performance
* Input-data quality

### 6. Cloud Deployment

Deploy the complete pipeline to cloud infrastructure such as:

* AWS
* Azure
* Google Cloud

### 7. Automated Retraining

Create a workflow such as:

```text
New Data
   ↓
Data Validation
   ↓
Retraining
   ↓
Evaluation
   ↓
Compare with Current Model
   ↓
Deploy if Better
```

---

# ⚠️ Limitations

This project is primarily designed to demonstrate **MLOps concepts and machine-learning workflow automation**.

Important limitations include:

* The model is trained on a fixed dataset.
* Model performance depends on the quality and representativeness of the dataset.
* No production-grade data validation layer is currently implemented.
* No production model-serving API is included in the main pipeline.
* No real-time model monitoring is implemented.
* No automated model deployment pipeline is currently provided.
* Model predictions should not be interpreted as certified laboratory water-safety results.

---

# 📖 Learning Outcomes

By studying this project, you can understand:

### Machine Learning

* Binary classification
* Random Forest
* Train/test splitting
* Missing-value treatment
* Accuracy
* Precision
* Recall
* F1-score

### MLOps

* Data versioning
* Pipeline versioning
* Reproducible training
* Experiment tracking
* Hyperparameter experimentation
* MLflow
* DagsHub
* DVC
* DVC Live
* Unit testing
* Linting
* CI concepts

---

# 👨‍💻 Author

## Piyush Singh

**B.Tech CSE | Applied Machine Learning**

VIT Bhopal University

---

# ⭐ Acknowledgement

This project was created as a hands-on exploration of **Machine Learning Operations (MLOps)** concepts, focusing on building reproducible and trackable machine-learning workflows rather than only developing a predictive model.

---

# 📄 License

This project includes a license file in the repository. Please refer to the repository's `LICENSE` file for the applicable licensing terms.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and exploring the different MLOps experiments included in the project.

```text
Data → DVC → Pipeline → ML Model → Metrics
                    ↓
             Experiment Tracking
             ↙       ↓        ↘
           DVC     MLflow    DagsHub
                    ↓
                 Testing
                    ↓
                   CI
```

**End-to-end MLOps for Water Potability Prediction 💧🤖**
