D:\projects\DVC project2>pip install cookiecutter
D:\projects\DVC project2>cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
D:\projects\DVC project2>cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
You've downloaded C:\Users\HP\.cookiecutters\cookiecutter-data-science before. Is it okay to delete and 
re-download it? [y/n] (y): y
  [1/8] project_name (project_name): water-potability-prediction
  [2/8] repo_name (water-potability-prediction): water-potability-prediction
  [3/8] author_name (Your name (or your organization/company/team)): data-thinkers
  [4/8] description (A short description of the project.): water potability prediction using ML
  [5/8] Select open_source_license
    1 - MIT
    2 - BSD-3-Clause
    3 - No license file
    Choose from [1/2/3] (1): 3
  [6/8] s3_bucket ([OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')): 
  [7/8] aws_profile (default): 
  [8/8] Select python_interpreter
    1 - python3
    2 - python
    Choose from [1/2] (1): 1
D:\projects\DVC project2>cd water-potability-prediction 
D:\projects\DVC project2\water-potability-prediction>python -m venv myenv
D:\projects\DVC project2\water-potability-prediction>myenv\Scripts\activate
(myenv) D:\projects\DVC project2\water-potability-prediction>pip list
(myenv) D:\projects\DVC project2\water-potability-prediction>git init

(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added Cookie Cutter Template"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
(myenv) D:\projects\DVC project2\water-potability-prediction>git remote add origin https://github.com/PiyushVIT346/water-test2.git
(myenv) D:\projects\DVC project2\water-potability-prediction>git push origin master
Add code of data_collection.py
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added data_collection.py"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add code of data_prep.py
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added data_prep.py"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add code of data_collection.py
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added data_collection.py"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add code of data_building.py
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added data_building.py"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add code of data_eval.py
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added data_eval.py"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add code of params.yaml
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "Added params.yaml"
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
Add Requirements
(myenv) D:\projects\DVC project2\water-potability-prediction>pip install -r requirements.txt
(myenv) D:\projects\DVC project2\water-potability-prediction>pip freeze > requirements.txt

(myenv) D:\projects\DVC project2\water-potability-prediction>dvc init
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc stage add -n data_collection -d src/data/data_colletion.py -o data/raw python src data/data_collection.py
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc repro
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc dag
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc stage add -n pre_processing -d src/data/data/data_prep.py -d data/raw -o data processed python src/data/data_prep.py
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc repro
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc dag
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc stage add -n model_building -d src/model/model_building.py -d data/processed -o models/model.pkl python src/model/model_building.py
Add this line in dvc.yaml in data_collection stage under dependency
params:
    - data_collection.test_size
Add this line in dvc.yaml in model_building stage under dependencies
params:
    - model_building.n_estimators
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc repro
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc stage add -n model_eval -d src/model/model_eval.py -d models/model.pkl --metrics reports/metrics.json python src/model/model_eval.py
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc repro
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc dag
(myenv) D:\projects\DVC project2\water-potability-prediction>dvc metrics show
(myenv) D:\projects\DVC project2\water-potability-prediction>git status
(myenv) D:\projects\DVC project2\water-potability-prediction>git add .
(myenv) D:\projects\DVC project2\water-potability-prediction>git commit -m "updated requirements.txt"
(myenv) D:\projects\DVC project2\water-potability-prediction>git tag -a v1.0 -m "Release V1"
(myenv) D:\projects\DVC project2\water-potability-prediction>git push origin master


water-potability-prediction
==============================

water potability prediction using ML

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
