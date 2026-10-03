water-potability-pipeline-experimenttracking

Add code of data_collectiona.py, data_prep.py,model_building.py, model_eval.py and params.yaml
start env
intialize git and dvc and cookiecuttter
pip intall dvclive
modify the eval code and add dvclive library

perform experiment code in terminal
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>dvc repro
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>dvc exp run
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>dvc exp diff funky-mime dedal-plow
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>dvc exp show
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>dvc exp run --queue -S data_collection.test_size=0.20,0.30,0.40 -S model_building.n_estimators=100,250,300  
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>git status
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>git add .
(myenv) D:\projects\DVC project2\water-potability-pipeline-experimenttracking>git commit -m "Done experiment"



==============================

water-potability-pipeline-experimenttracking

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
