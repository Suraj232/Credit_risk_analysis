# Credit_risk_analysis
# Overview
The main purpose of this analysis is to use Machine Learning algorithms to make predictions on the given data. This challenge is mainly focused on Supervised learning which includes a labeled outcome.

The process of Machine learning goes from fitting and training the model, and evaluate the data. The dataset from LendingClub has three classification in Loan_status where Issued status was taken out of the dataset. And other two classification are low_risk and high_risk. To have a better study with Machine learning we need to resample the data using different algorithms. Some of the algorithms used are SMOTE, ClusterCentroids, SMOTEENN, RandomOverSample, BalancedRansomForestClassifier, and EasyEnsembleClassifier.

# Results
The data provided is resampled by using Machine Learning and python libraries: scikit-learn and imbalanced-learn.
The dataset is cleaned and the loan_status is classified in to low_risk and high_risk. Once the classification factors are determined the available data is reduced to 68,817.

<img width="237" alt="total_Data" src="https://user-images.githubusercontent.com/110261837/209597574-c48fbe4d-a0ee-41e8-8bb3-4d4031123707.png">

### 1.) Oversampling
In RandomOverSampler model collects random sample untill the sample size are equal in both ends. 

<img width="237" alt="total_Data" src="https://user-images.githubusercontent.com/110261837/209599590-966efeed-39ca-4766-a460-237e0cccf6a7.png">
- Balanced Accuracy Score: 64.85 %

The classification matrix is applied to get the result of oversampling model. The result is shown below:
- The precision rate is 1% for High_risk and F1 score is at 2%.
- The precision rate is 100% and F1 score is at 83% for Low_risk.

<img width="507" alt="Class" src="https://user-images.githubusercontent.com/110261837/209600108-a27d1b48-c1b6-4c89-8daa-c116d7f7cb30.png">

### 2.) SMOTE 
SMOTE is like RandomOverSample increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.
The accuracy score is 61.98%.

<img width="401" alt="Smot" src="https://user-images.githubusercontent.com/110261837/209607055-56ca4d0b-ede0-461e-beca-0bde00fda370.png">

The Classification matrix is applied to get the results for SMOTE. The result is shown below:
- The precision rate is 1% for High_risk and F1 score is at 2%.
- The precision rate is 100% and F1 score is at 82% for Low_risk.

<img width="508" alt="SmotClass" src="https://user-images.githubusercontent.com/110261837/209607622-36c0e6b4-070f-49b2-9956-9964532f92ec.png">

### 3.) UnderSampling
ClusterCentroids Model, is an algorithm that measures cluster of the majority class to generate synthetic data thats representative of the cluster. 

<img width="299" alt="clustercounter" src="https://user-images.githubusercontent.com/110261837/209608159-26523295-49a0-4431-9089-84abbb0cba94.png">

The accuracy score is 48.91%.

The Classification matrix is applied to get the result for Undersampling.
- The precision rate is 0% for High_risk and F1 score is at 1%.
- The precision rate is 100% and F1 score is at 66% for Low_risk.

<img width="526" alt="Classcounter" src="https://user-images.githubusercontent.com/110261837/209608438-43c00c52-9e52-4fa4-af3d-10e4f7fde3f1.png">

### 4) Combination Sampling
SMOTEENN Model combines aspects of both oversampling and undersampling. 

<img width="323" alt="combicounter" src="https://user-images.githubusercontent.com/110261837/209608777-4b2559ba-c703-4f32-a6b2-852e4d0f36e7.png">

- The accuracy score improved to 64.16%. 

- The precision rate is 0% for High_risk and F1 score is at 2%.
- The precision rate is 100% and F1 score is at 74% for Low_risk.

<img width="516" alt="combiclassr" src="https://user-images.githubusercontent.com/110261837/209608942-52e94f9c-b2f3-4072-af25-5f6e9edd5760.png">

### 5) Ensemble Classifiers
BalancedRandomForestClassifier, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.
- The balanced accuracy score increased to 78.85% for the model.
<img width="318" alt="Ensemble" src="https://user-images.githubusercontent.com/110261837/209611431-ef54214e-5dcd-4dd3-90e6-6ab7102314f6.png">

- The High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
- Low Risk still had a precision rate of 100% with the recall at 87%.
- The top feature by importance was "total_rec_prncp" at 7.9% of the total.

<img width="532" alt="Ensembleclas" src="https://user-images.githubusercontent.com/110261837/209611815-8ab50ddc-fc43-4c65-9524-b2d8f9fb9d95.png">

### 6) AdaBoost Classifier
EasyEnsaembleClassifier is a set of classifier where individual decisions are combined to clasify new example.

- The balanced accuracy score increased to 93.2% with this model.
<img width="252" alt="Adaaccuracy" src="https://user-images.githubusercontent.com/110261837/209615264-666fe1c3-e541-430f-89f9-bba1b4e31cbc.png">


- The High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
- Low Risk still had a precision rate of 100% with the recall at 94%.

<img width="571" alt="Adaclass" src="https://user-images.githubusercontent.com/110261837/209615424-33203d2f-eb31-419f-aa71-3fa3692d367a.png">


# Summary
Out of all six model, the EasyEnsaembleClassifier has the best result with an accuracy rate of 93.2% and a 9% precision rate when predicting Risky candidates. 

### The result according to the accuracy related to high risk:

- EasyEnsaembleClassifier: 93.2% accuracy
- BalancedRandomForestClassifer: 78.9% accuracy
- RandomOverSample: 64.85% accuracy
- SMOTEENN: 64.19% accuracy
- SMOTE: 61.98% accuracy
- ClusterCentroids: 48.9% accuracy




