Add the code to water_potabilty
enter this command in terminal
D:\projects\DVC project2>mlflow ui

add new terminal

run code using command: python .\water_potability.python

open mlflow ui website
click on the parent model created 
click on register model button on top right button
click on create new model option 
enter the name "water_potability_rf" as model name and click on Register button 
Hence the first version is created 
click on models tab to see the model that has been register and look the stages information 
on right top click on stage None option and choose Transition to Stagging option . We can choose other option also to show that whether it is in staging,production stage,deployement stage.

if any changes is done in code to improve the result. We need to change the code and perform steps again. 
a new version is been created. We need to register it. While chossing the name of register model than select the name of already registered model, so it form it's new version.
