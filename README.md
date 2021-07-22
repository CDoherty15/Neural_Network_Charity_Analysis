# Neural_Network_Charity_Analysis
Neural Network application

## Overview
We are using our knowledge of machine learning and nueral networks to create a binary classifier that will be able to predict whether applicants will be successful if they are funded by Alphabet Soup. Alphabet Soup has provied a dataset containing over 34,000 organizations that have been funded from over the years. It is up to us to preprocess the data, which involves finding the target(s), finding the features, and droping unnecessary columns. The columns provided are:
- `EIN` and `NAME`: identification columns
- `APPLICATION_TYPE`: Alphabet Soup application type
- `AFFILIATION`: Affiliated sector of industry
- `CLASSIFICATION`: Government organization calssification
- `USE_CASE`: Use case for funding
- `STATUS`: Active Status
- `INCOME_AMT`: Income classification
- `SPECIAL_CONSIDERATIONS`: Special Consideration for application
- `ASK_AMT`: Funding amount requested
- `IS_SUCCESSFUL`: Was the money used effectively

## Results
### Data Preprocessing

***Target varibale***
- The goal of this neural network is to classify whether or not an can be successful through funding by Alphabet Soup. We are presented with organizations from the past and the `IS_SUCCESSFUL` is provided. This column is already binary, meaning it only has 2 unique values: "0" for unsuccessful and "1" for successful. We will make this column our target variable and since it is already in a binary format, we will not have to encode it. 

***Remove unneccessary columns***
- We can only use numerical data in the neural network and a couple of these columns are object types. So we can either encode these columns or we can drop them. However, dropping variables is very risky since omitting any data can skew the accuracy and results, especially if the reason is just because the values are not numerical. Looking at the descriptions, a lot of these columns are crucial in determining if they organization was successful. But, the identification columns, `EIN` and `NAME` are just simply the EIN number and the name of the organization. These are not necessary for the analysis and therefor we can drop these without effecting the data.

![EIN_NAME_cols](https://user-images.githubusercontent.com/79118630/126688385-719b8628-4209-4bb7-90db-4e0aa9407474.png)

***Feature Variables***
- We started with 11 columns, we have dropped `EIN` and `NAME`, and we know `IS_SUCCESSFUL` is our target column. Now, the remaining 8 columns will be our feature columns. Next we need to figure out if we need to bin any of these features by looking at how many unique values they have: 

![feature_unique_values](https://user-images.githubusercontent.com/79118630/126688985-b8bc2201-b4b9-4190-b1ba-d1617567fd7d.png)
- Typically, any feature that has over 10 unique values will be needed to be binned. We will only need to look closer at `APPLICATION_TYPE` and `CLASSIFICATION`. `ASK_AMT` is the amount that each organization asked for, so we do not need to encode this since that could skew that data since each value is unique to the organization.

_APPLICATION_TYPE_ 
- First we will look at `APPLICATION_TYPE`. We will need to encode the column but first will need to set a limit so that we can bin certain values so that we have less feature columns so the neural network can run more efficiently and smoothly. We can do this by looking at the value counts for each unique value or by plotting the density of the column. Here we will do both:

![APP_TYPE_uniquecount](https://user-images.githubusercontent.com/79118630/126691527-e1dc6172-bf84-4f96-8426-b5b6a9f464f0.png) ![APP_TYPE_density](https://user-images.githubusercontent.com/79118630/126691536-4fcc182b-aa2c-43bf-a418-3f4bcc2896e0.png)
- We can see by the plot that the dip between 0 - 10,000 is very steep, so it is hard to get a good estimate on where we should set the cutoff point. If we look at the unique value counts, the highest has over 27,000 and the next few are all lower than 2,000. If we meet in the middle of the plot and chose 5,000, we would bin all of the application types except for one. This is why it is good to do both. Looking at the counts, there is a big difference from "T10" to "T9", 528 to 156. So we will set our cutoff point at 500. We have 17 unique application types, after setting our cutoff point to 500, we are now left with 9 unique application types, 8 of the original and the 9th is the binned type.

_CLASSIFICATION_
![CLASS_unique_valuecount](https://user-images.githubusercontent.com/79118630/126692962-e5bd05d4-7a20-4d3b-a1b1-f9836819a5e7.png)![CLASS_density](https://user-images.githubusercontent.com/79118630/126692976-83fe8f16-e779-400f-9345-591f9e22c093.png)

- We conduct the same process with `CLASSIFICATION`. This one is a bit more challenging since there are 71 unique values in this column. On the density plot, the drop is very short but is also very steep. We look at the value counts and there is another substantial differences between the counts. Not only that, but it seems that a lot of the unique values are only in the dataset once. So here, we can play it safe and make the cutoff point 1800. We started with 71 unique values, and now are left with 6, 5 of the original and 1 'other' bin.

#### Final Preprocessing Step
- Now we can encode the columns that need to be encoded: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `SPECIAL_CONSIDERATION`. We create a dataframe of just the encoded variables, which gives us 41 columns. We then merge these with the 3 columns that didn't need to be encoded: `STATUS`, `ASK_AMT`, and `IS_SUCCESSFUL`. Now we have a dataset with 44 columns, with 1 target and 43 features and is now ready to be split into train and testing sets and ready to be ran through a neural network. Here is a list of all the columns:

![application_df_cols](https://user-images.githubusercontent.com/79118630/126694838-af3584e1-dce0-4aa0-8aef-344d88b2d992.png)

### Compiling, Training, and Evaluating the Model

## Summary
