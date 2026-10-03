install mlflow
pip install mlflow

to see gui of mlflow
mlflow ui

create a new terminal 
D:\projects\DVC project2>cd Experiment_using_mlflow
let value of n_estimators=500
D:\projects\DVC project2\Experiment_using_MLFLOW>python .\src\water_model.py
now check it on mlflow website
change value of n_estimators=1000
D:\projects\DVC project2\Experiment_using_MLFLOW>python .\src\water_model.py
now again check mlflow website

creating new Experiment 
add new py file "water_model_gb.py"
The code is same of earlier experiment, just use gradientboosting instead of randomforest
create a new experiment on mlflow website and name it water_exp2 
add this line in code: mlflow.set_experiment("water_exp2")
now at different n_estimators value check the metrics on mlflow

adding artifacts
add these line in code: 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("http://127.0.0.1:5000")

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")
Check it on mlflow website 

To get some artifact files(MLmodle,conda.yaml,model.pkl,python_env.yaml,requirements.txt) in mlflow website
add these lines:
import mlflow.sklearn
mlflow.sklearn.log_model(clf,"GradientBoostingClassifier")

To track the code(water_model_gb.py) using artifact add this code:
mlflow.log_artifact(__file__) 

To add tag to code add this code:
mlflow.set_tag("author","data_thinkers")
mlflow.set_tag("model","GB")