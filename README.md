# IRIS DATA PROJECT

In this project, Iris data is used. Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters.

Main purpose of the project is predict the species of the Iris by checking the sepal length, sepal width, petal lenght and petal width. **Logistic Regression, KNN, SVM and Random Forest Classification** models are used.

For the deployment part, FastAPI is used. The HTML page provides the basic code required to load the data. Please, open terminal and type block code directly or open the prediction.py file then follow the below steps:

- `uvicorn prediction:app --host 0.0.0.0 --port 76 --reload`   or
- `uvicorn prediction:app --host localhost --port 8080 --reload`

# Sample Image via using FastAPI
<div align=center>
<img src="https://i.ibb.co/1r8W0nK/download.png" alt="Iris" width="95%" height="100%">

</div>
