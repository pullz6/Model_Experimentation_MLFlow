# Model Experimentation with MLFlow

This is a beginner-friendly project to understand how to create experiments and test model versions a simple model with MlFlow locally. The data for this project is available here https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data. 


## Table of Contents

1. [Project Overview]
2. [Installation]
3. [Project Structure]

---

## Project Overview

### About

Our objectives are the below:

- Testing a bagging regressor model with MLFlow.
- Creating a streamline process for data ingestion, data preparation, model training and model evaluation. 
- MLFlow, Sklearn, Numpy and Pandas are mainly used, few other ancillary libraries are used as well. 

### Screenshots or Demos

Below screenshot of the logged model after pre-processing, training and testing steps completed with MLFlow. 

![image](https://github.com/user-attachments/assets/874b5439-394f-411a-9537-c11c9b3fbad0)
![image](https://github.com/user-attachments/assets/bb09147b-e247-4d96-a9df-7403061d4c0a)


## Installation

### Prerequisites

Below libraries needs to be installed in the virtual environment: 

1. MLFlow
2. Pandas
   

### Instructions

After cloning the repository, please follow the below steps:

1. Install all the libraries mentioned above into the virtual environment. 
2. Afterwards, run 'python tracker.py'
3. Finally, run 'mlflow ui --port 5000' to view the MLFlow dashboard to view the logged model.
4. If you improve your model, you can re-run the steps again and see the improved model. 

---

## Project Structure

```
├── loading_data.py      # Ingests the data from the Kaggle website. 
├── cleaning_data.py     # Cleans the data and prepares it. 
├── model.py             # The model architecture, training and evaluation. 
├── tracker.py           # Logging all details into MlFlow for the training run and inference. 
└── README.md            # Project documentation

```

---




