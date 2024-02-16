# Health Insurance Cross-Sell Prediction
Supervised Ml Classificaton Project

Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

## EDA Observations
there are 206089 Male customers and 175020 Female customers
there are 380297 people who own Driving License and 812 don't
It is observed that people who have previously not insured are intrested in the policy, so it is better to tap the market who previously have not insured
It is observed that most of the people with vehicle age is 1-2 years. ratio wise people with vehicle age greater than 2 years are more intrested in buying policy.
![image](https://github.com/GeetanjliRC/HealthInsurance-CrossSellPrediction/assets/91873936/f2cbd256-f83c-4276-a9df-05b641460ad2)

Its found that previously customers who got their vehicle damaged are more intrested in buying new policy.
![image](https://github.com/GeetanjliRC/HealthInsurance-CrossSellPrediction/assets/91873936/48a68b18-dc73-495c-86aa-cc44d3711b75)

Most of the premium falls under range 5000-100000
![image](https://github.com/GeetanjliRC/HealthInsurance-CrossSellPrediction/assets/91873936/85835cf6-9659-4c41-8546-bc5c6a1b8bf1)



## Project Summary

We performed hyperparameter tuning on various models using cross-validation:
GridSearchCV is used to perform a grid search over the specified hyperparameters.
cv=5 specifies a 5-fold cross-validation. The hyperparameter that gives the best cross-validation score is selected as the optimal hyperparameter for the model.

The feature 'Previously_Insured_yes' can be considered as most important with relative importance of 0.38.
The next 4 features are vintage, annual_premium, age and vehicle_damage-yes can be considered with relative importance 0.26, 0.125, 0.095, 0.075 respectivly
As these 5 main features play a role in decreasing the value of entropy, the machine learning model, random forest classifier considers them closer to the root node.

The insurance company can deploy a machine learning model(Tuned with best parameters) that uses XGB Classifier to predict the wheather the already existing health insurance customer would be interested in a vehicle insurance product. The company can improve the conversion rate by taking steps to encourage people to buy vehicle insurance by offering some incentives/ease of application & claim settlement process.
