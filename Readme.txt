In CS2 session we followed below steps
1) installed miniconda
2) created mlops miniconda env
3) trained a sample iris model (iris_classification_for_mlops)
4) using joblib we serialized the above model (code at the end of iris_classification_for_mlops)
5) then using flask we created a rest api to interact with the model (app.py)
6) In one terminal session ran the app.py 
7) Opened another new terminal and used the command and successfully tested the API (command_test_flask_application)