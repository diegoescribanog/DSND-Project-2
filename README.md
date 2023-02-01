## Udacity's Data Scientist Nanodegree - Project 2: *"Disaster Response Pipeline"*

### Table of Contents

1. [Installation](#installation)
2. [Project motivation and components](#motivation_components)
3. [Repository's structure](#structure)
4. [How to run](#how_to_run)
5. [Example of output](#output)

## Installation <a name="installation"></a>

The project utilizes the following Python libraries included in the Anaconda distribution:

- The fundamentals: Pandas and Numpy. 

- ML libraries: Sklearn and XGBoost. 

- NLP libraries: NLTK and Gensim. 

- ML model saving and loading: Joblib. 

- Static visualizations: Matplotlib and Seaborn. 

- Interactive visualizations: Plotly. 

- SQLite Databases: SQLalchemy. 

- Web deployment: Flask. 

Additionally, it is necessary to install the library Iterstrat ([more info here](https://github.com/trent-b/iterative-stratification)), to perform stratified train-test split on the multi-label dataset. 

The code should run with no issues using Python versions 3.*.

## Project motivation and components <a name="motivation_components"></a>

This project is part of the requirements for Unit 2 in Udacity's Data Scientist Nanodegree. 

The project involves utilizing data from [Figure 8](https://appen.com/) to develop a model for a disaster message classification API. The dataset contains actual messages sent during various disaster events (such as earthquakes, hurricanes, and floods) and the objective is to build an ETL and ML pipeline to categorize the different messages and forward them to the relevant disaster relief organization, to facilitate automation and speed up the process.

The project consists of the following components:

1. ETL pipeline
	- Loads and merges datasets. 
	- Cleans the data. 
	- Stores it in a SQLite database. 
2. ML pipeline
	- Loads data from the SQLite database. 
	- Splits the dataset into stratified training and test sets (for multi-label data).
    - Trains and tunes a model using GridSearchCV. 
    - Outputs results on the test set.  
	- Builds a text processing and machine learning pipeline. 
	- Exports the model as a pickle file. 
3. Flask Web App



## Repository's structure <a name="structure"></a>

The repository is structured as follows:

~~~~~~~
        DSND-Project-2
          |-- app                            
                |-- templates                
                        |-- go.html
                        |-- master.html
                |-- run.py                   
          |-- data
                |-- messages.csv     
                |-- categories.csv   
                |-- DisasterResponse.db      
                |-- data_preparation.py  
                |-- ETL Pipeline Preparation.ipynb  
          |-- models
                |-- classifier.pkl    
                |-- train_classifier.py  
                |-- multilabel_split.py 
                |-- ML Pipeline Preparation.ipynb  

          |-- Screenshots_example 
                |-- example_1.png
          |-- README
~~~~~~~


## How to run<a name = "how_to_run"></a>

To correctly deploy the web app, these steps must be followed:

1. Move to the project's root directory. 

2. Run the ETL pipeline to load the raw data, clean it, and store it in a SQL database. 

```python 
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
```

3. Run the ML pipeline that trains the model, evaluates it and saves the final version. 

```python 
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 
```

4. Move to the app's directory.  

5. Run the app.
```python 
python run.py
```

6. Open [http://0.0.0.0:3001/](http://0.0.0.0:3001/) and use the app. 

## Example of output <a name="output"></a>

The final version of the app is very intuitive and easy to use. As an example, the following image represents the output of the app for the message *"we need water and food"*. 

![example output](./Screenshots%20example/example_1.png)
