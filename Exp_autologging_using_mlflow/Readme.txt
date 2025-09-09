add these lines in code
train_df=mlflow.data.from_pandas(train_processed_data)
test_df=mlflow.data.from_pandas(test_processed_data)
mlflow.log_input(train_df,"train")
mlflow.log_input(test_df,"test")

start mlflow ui
D:\projects\DVC project2\Exp_autologging_using_mlflow>mlflow ui 

now start other terminal.

run the code: D:\projects\DVC project2\Exp_autologging_using_mlflow>python .\dataset_train.py      

for autologging, create a new file "water_autologging.py"
just remove the code for manual logging and add this code: 
mlflow.autolog()

just now run the code:
python .\water_autologging.py
now check the result on mlflow website 


