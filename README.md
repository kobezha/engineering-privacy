# Enhanced Privacy in Health Devices Implementation

In this project, we investigate the topic of privacy in health devices and explore two possible techniques that can help us enhance the level of privacy in health devices. Specifically, we explore dataset de-identification through k-anonymity data processing and a more secure edge-device machine learning approach through Federated Learning.

The primary tool used for implementing federated learning is the [Flower Framework with TensorFlow](https://github.com/adap/flower) 

# Dataset
The dataset used for this project is the Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It can be found [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). All patients within the dataset are females at least 21 years old of Pima Indian heritage. The purpose of the dataset is to predict whether or not a patient has diabetes based on diagnostic measurements. These 8 predictor variables include the number of pregnancies the patient has had, plasma glucose concentration, blood pressure, skin thickness, their BMI, insulin level, age, and Diabetes Pedigree Function. The target variable is a binary outcome indicating if the patient has diabetes or not. Within this dataset, we considered the outcome variable as being sensitive.

# How to setup and run the project

# Credits
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
