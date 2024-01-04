#!/usr/bin/env python
# coding: utf-8

# # Introduction to  Machine Learning-1

# # Q1: Explain the following with an example:
# 1) Artificial Intelligence
# 2) Machine Learnin,
# 3) Deep Learning
# 
# Artificial Intelligence:

->In my understanding Artificial Intelligence is used in order to create Smarter Applications that comprises of 4 main pillers in order to create them:
  1.Data
  2.Training 
  3.Engine
  4.Action/Deployment
  
  The application Created upon these pillers perform its own task without the human interventions.
  Examples: Self Driving Cars,Robots,Alexa etc.
>It is an umbrella term which encompassing diverse approaches like machine learning, deep learning, natural language processing, computer vision, and robotics.  
--------------------------------------------------------------------------------------------------------------------------------  
# Machine Learning

->It is a wide field that provide stats tool to analyze, visualize,predictive model,forecasting.
->It consist a wide range of algorithms eg: Linear and Logistic Regression,SVM and Decision Tree,KNN and CPA and many more all     the algorithm have their specific purpose and use.

->Moving forward with the type of Machine Learning.
  1.Supervised Machine Learning:
      1.1) Classification : here the output will be the categorical in nature.
      1.2) Regression: here the output will be continuous in nature.
  2.Unsuprevised Machine Learning:
      - Clustering concept is to be used here.
  3.Semi-Supervised Machine Learning:
      - Combination of Supervised+Unsupervised Machine Learning
  4.Reinforcement Learning:
         -here the intelligent agents ought to take actions in the environment to maximize the notation of Cumulative reward.
>Some example: Amazon.in,Netflix Both of them uses Recomendation System .         
--------------------------------------------------------------------------------------------------------------------------------

# Deep Learning:

->It basically is the subset of Machine Learing.
->It Mimic the Human Brain.
->Uses the Multi Layered Neural Network.
->Eg: Object Detection Image Recognition ,CHATBOT Recommendation System.


So as per my knowledge the basic term AI,ML,Deep Learning i have illustrate above.
      
  
  
# # Q2: What is supervised learning? List some examples of supervised learning. 
-> It is a part of Machine Learning in which the output feature of the dataset is known and the modles are tarining upon the training datasets.
-> It is basically divided into 2 parts:
    1.Supervised Machine Learning:
      1.1) Classification : here the output will be the categorical in nature.
      1.2) Regression: here the output will be continuous in nature.
      
->Examples:

   1) Classification:
    Let we have the following dataset:
    No. of hour played : 8,7,6,5,4
    No. of Study hours : 2,3,4,5,6    
    So based upon the above dataset we have to predict menas we have to give the output feature as Pass/Fail which is
    categorical in nature.
    Pass/Fail: Fail,Fail,Fail,Pass,Pass.
    And this output feature is totally dependent upon our given input.
    
   2) Regression:
    Let we have the following dataset:
    Size of house :
    No. of Rooms  :
    So based upon the input feature will be predicting the Price of the house which is our output feature and which is 
    dependend upon the input feature.
    
    
# # Q3:  What is unsupervised learning? List some examples of unsupervised learning.
-> Unsupervised learning is a technique in which models are not supervised/trained using training dataset. Here the models itself find the hidden patterns and insights from the given data.
-> Here the concept of Clustering is to be introduced.
-> Clustering simply means grouping the information of similar data.

->Example:
 1.Let suppose we are taking an example of Customer segmentation where the customer are divided upon the basis of expenditures 
 and salary,so that the 
    -clustors having salary high and expenditure low  lies in the top left part of the graph
    -clustors having salary low and expenditure is quite high  lies in the bottom left part of the graph
    -clustors having salary high and expenditure high  lies in the top right part of the graph
    -clustors having salary low and expenditure high  lies in the bottom right part of the graph
  
  2.It can also used in the image detection where a rich dataset of some animals are to be provided to the model and then the model in itself find thefeatures and based upon the features it identify images.
  
  3.In my opinion we can include Snapchat here like the way this application detect the facial nature of human and based upon that it applies the filters.
# # Q4: What is the difference between AI, ML, DL, and DS?
# AI: Ai simply is the smarter application that is created upon the basis of the understanding of ML,DL,DS.Applications that can perform its own tasks without any human intervension.

#ML: Subset of AI,ML provide all the statistical tools to analyze,visualize predictive model,forecasting.

#DL: Subset of ML,Its is something that mimics the Human Behavious which uses Multilayered NeuralNetwork that incorporate intelligence into machines.

#DS :Ds Uses Statistics,Linear Algebra,Calculus,Probablity in order to analyse and predict the behaviour of the model.
# # Q5: What are the main differences between supervised, unsupervised, and semi-supervised learning?
Main Differences are:
    Supervised ML:
        -Supervised learning algorithms are trained using labeled data.
        -Here,input data is provided to the model along with the output.
        -Supervised learning needs supervision to train the model.
        -Supervised learning can be categorized in Classification and Regression problems.
    Unsupervised ML:
        -Supervised learning algorithms are not trained using labeled data.
        -Here,Only input data is provided to the model.
        -It supervise itself to train the model.
        -Unsupervised Learning can be classified in Clustering and Associations problems.
    Semi-Supervised ML:
        -Combination of both means , Supervised + Unsupervised learning.
        -Trained on both labeled and unlabled data.
        
# # Q6: What is train, test and validation split? Explain the importance of each term.
->The train-test-validation split is fundamental in machine learning and data analysis,particularly during model development.
->It involves dividing a dataset into three subsets:
    Training, Testing, and Validation.
    
  Now let's se them one by one:
  
  #Training:
  -Here we will train our model through training dataset.
  #Validation:
  -Here the hyper tuning of the model is to be done like by increasing or decreasing the parameters the hypertuning is to
   be checked.
  #Test:
  -Model is finally is tested inorder to check its functionality.
  
Note: Here Model only have the info of trainng & Validation Dataset ,Testing Dataset is always hidden.  
# # Q7: How can unsupervised learning be used in anomaly detection?
Anomaly detection is one of the main application of Unsupervised learning which is basically used to identifying normal patterns within a data sample and then detecting outliers based on the natural characteristics of the data set itself.

Examples of Unsupervised Anomaly Detection Applications:

Fraud detection: Identifying unusual financial transactions or credit card activities.
Network intrusion detection: Spotting suspicious network traffic patterns indicative of cyberattacks.
Machine condition monitoring: Predicting equipment failures based on sensor data deviations from normal operating conditions.
Quality control in manufacturing: Detecting defective products during production based on anomalies in sensor readings.
# # Q8: List down some commonly used supervised learning algorithms and unsupervised learning algorithms.
# Supervise Learning Algorithm:
-> Decision Tree:
        It is basically a tree-like flowchart that maps choices and their potential outcomes, guiding decisions through a 
        series of yes/no questions.
->Logistic Regression:
    Predicting probabilities via a squished linear function that models log-odds.
->Linear Regression:
     Fitting a straight line to data to model relationships and make predictions.    
->Support Vector Machine:
    Finding the widest street between point-groups in high-dimensional space.

# UnSupervise Learning Algorithm:
->K-means clustering:
    Grouping data points into K clusters by iteratively minimizing distances to cluster centers.
->Hierarchical Clustering:
    Progressively merging or splitting data points to build a tree-like structure of clusters.    

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




