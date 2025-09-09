create a folder mlflow_exp_dagshub and add a file name water_train.py 
write code in it
push it into git 
go to dagshub website and login with github and connect the repository containing this code 
run the code using command D:\projects\DVC project2\Experiment_using_Dragshub\mlflow_exp_dagshub>python .\water_train.py
now reload dagshub and it will show the experiment result
click on open with mlflow to see the experiment result on mlflow

now it can be used by other person for experiment 
just clone the github 
D:\projects\DVC project2\Experiment_using_Dragshub\ml_flow2>git clone https://github.com/PiyushVIT346/mlflow_exp_dagshub
now make a new file also but in different directory
so created a new file in mlflow2/mlflow_exp_dagshub
also added the new experimental file water_train2.py having same format of original file 

now run the new experimental file water_train2.py using command:
D:\projects\DVC project2\Experiment_using_Dragshub\ml_flow2\mlflow_exp_dagshub>python .\water_train2.py
now check dagshub and mlflow website for new result been added there. 


https://dagshub.com/PiyushVIT346/mlflow_exp_dagshub.mlflow

import dagshub
dagshub.init(repo_owner='PiyushVIT346', repo_name='mlflow_exp_dagshub', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)