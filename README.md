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

***Compiling and Training***
- For our model, we used 2 hidden layers, the first hidden layer has 80 nodes and the second has 30 nodes. For both layers, we used a 'relu' activation as this seemed the best fit for our data. We then added an output layer that is 1 unit and is a 'sigmoid' activation. Here we have our neural network summary with the shape of each layer and the number of parameters. 

![origin_nn_summ](https://user-images.githubusercontent.com/79118630/126696156-ddd40f77-33a0-44b9-97a0-7dddfa5de8d4.png)
- Next we compile the model so that the loss is measured by 'binary_crossentropy' and we are measuring the model my an accuracy metric. We then fit and train the model with the training sets and run it for 100 epochs. 

***Evaluating the model***

Here are the results for the model:

![origin_nn_results](https://user-images.githubusercontent.com/79118630/126696775-aa1e8471-e95c-4187-932e-64356cee7d85.png)
- We ended our results with a 0.5561 loss and an accuracy of 72.56%. This is accuracy is not bad but it is not very good either. Loss metric is the score of the performance of the model through each iteration and epoch by evaluating the inaccuracy of a single input. Our accuracy is low, but the bigger problem is the high loss. This loss means that there is a 55.6% inaccuracy for each single input. This is just a factoid to keep in the back of our heads because we are mainly focused on the accuracy. 
- For this project, we want our accuracy to be 75% or greater. We will give ourselves three attempts to change our neural network model in trying to reach 75% accuracy. 

#### Optimizing 
- For this optimization process, we will follow the first preprocessing steps. For our first attempt, we will change the binning size for `APPLICATION_TYPE` and `CLASSIFICATION`

***First Optimization***
- Originally, we had set the `APPLICATION_TYPE` cutoff to 500 and `CLASSIFICATION` to 1800. This time `APPLICATION_TYPE` will have a cutoff of 100 and `CLASSIFICATION` will have a cutoff at 600. Changing the bin size allows for more of the original values to be encoded, allowing for more accuracy because less values will be the raw values and not binned in 'other'. These changes don't seem large since only one or two of the original unique values are used again, but the 'other' bins are much lower:

![APP_TYPE_valuecount_OPT1](https://user-images.githubusercontent.com/79118630/126698452-4d2b09ed-b134-41fe-82ef-fd763584b09c.png) ![CLASS_valuecount_OPT1](https://user-images.githubusercontent.com/79118630/126698461-9e2e82f3-d3cc-45ee-a696-0a027e8064b0.png)
- We follow the same encoding steps, merge the columns that didn't need to be encoded, and now have a total of 46 columns. 1 target column and 45 feature columns. Adding features doesn't always increase accuracy, but we technically didn't add features, we just put more of the original data back in the dataset.
- We then split the data like before in training and testing, and we will use the same hidden layers, nodes and activation functions as we did with the original model.

![OPT1_nn_summ](https://user-images.githubusercontent.com/79118630/126699367-dc82afb7-23b9-4c8d-98e7-0cc42f307da3.png)
![OPT1_nn_results](https://user-images.githubusercontent.com/79118630/126699537-58589595-038a-41e8-a483-bfee2be5f79f.png)

- Even though we didn't change anything to the model, we get an increased amount of parameters because we added 2 feature columns. Our results are pretty much the same, but slightly better with a loss of 0.5522 and an accuracy of 72.7%. We failed in trying to increase our accuracy to 75% and our increases were microscopic

***Second Optimization***
- For our second attempt, we will add another hidden layer with 10 nodes. Adding another layer should improve our model because it is another cycle of processing the data will have to go through. We will also keep our new bins because that has shown to better our model. 

![OPT2_nn_summ](https://user-images.githubusercontent.com/79118630/126700051-e876346c-4f47-4b0c-9499-6499338317b6.png)
![OPT2_nn_results](https://user-images.githubusercontent.com/79118630/126700070-b8b708c0-a62d-4c7b-9088-2bfca1ce9f7e.png)
- The parameters for the first two layers stay the same cause we did not touch those, but we increase the total amount of paramters because we added another hidden layer. However, adding the hidden layer did the opposite of what we wanted. We got a loss of 0.5545 and an accuracy of 72.45%. We have one last attempt to reach our 75% accuracy goal. 

***Third Optimization***
- For our final attempt, we will keep everything the same from the first two optimization attempts and this time instead of running epochs, we will run 200. Epochs are how many times the data is ran through the model and increasing that count usually helps because that is increasing the training. We will keep our third hidden layer because even though our loss and accuracy did worse compared to first optimization attempt, the loss did better than the original model and the accuracy was only marginally worse (.11%). So running more epochs through could potentially worsen our accuracy, but it could increase it. 

![OPT3_nn_summ](https://user-images.githubusercontent.com/79118630/126700887-13aa1962-75e6-4057-bd48-810b448bdc52.png)
![OPT3_nn_results](https://user-images.githubusercontent.com/79118630/126700895-667ece58-2940-4097-99b9-f94e1797ee11.png)
- Our summary table will stay the same cause we didn't change any of the layers. Our results however showed little change. The loss was 0.5562 and the accuracy was 72.63%. Compared to the second attempt, our accuracy actually increased but our loss decreased, but these changes again are so microscopic that if we are too make more attempts, we will need bigger changes. 

## Summary
- Our first goal was to create a neural network in order to determine if an organization can be successful or not when funded by Alphabet Soup. We created a neural network that provided around 72% accuracy, which is substantial but not good enough. So we needed to optimize the model to see if we can increase the accuracy to 75%. 
- Our first attempt was to change the binning size for our two features that needed to be binned, and the result was probably the best one. Loss went down (which is a good thing because that means lowers percentage of inaccuracy) and our accuracy went up. Application type already had 9 unique values after the second binning, so maybe next time decrease the cutoff point for classification and that could help. We can summize that binning the original data too much can lower our accuracy and loss metrics.  
- The second attempt involded adding a hidden layer with 10 nodes. The loss and accuracy did worse compared to the first attempt, but compared to the original, only the accuracy did worse. Therefore we can say that when we add a hidden layer, the loss of a model lowers. Meaning, the percentage of inaccuracy for every input decreases, which may not help the overall accuracy, but the algorithm reads the data more clearly. 
- The final attempt used the first two attempts and increased the epochs, which increases how many times the data is ran through the model. Compared to the second attempt, the third had slightly better accuracy but a higher loss. However, compared to the original, the third attempt was very similar. The third attempt's loss was only 0.0001 worse than the original and was 0.07% better.
- My biggest recommendation for this dataset would be to fine-tune the feature columns and to run more than 100 epochs (or at least 200). When we changed the bin size, both the loss and accuracy improved and running more epochs increased the accuracy. Adding a hidden layer only lowerd the accuracy by a little bit but not tremendously to the point where adding a layer will be bad, but it something to lookout for. 
- I would say one change would be to change the activation functions. The model had very close accuracy to what we wanted, but it was right on the cusp. The problem with large datasets and big neural networks is that these take time to run. Luckily, this data only took a few minutes, so trial and error will work with this project. All of the optimization attempts usually effect the results, maybe not significantly but they will change. They did change but notthing noteworthy. In fact, if we round the results the loss will sit at 55-56% and accuracy is 72-73%. So it seems that one of the big changes that will be needed will be to change the activation functions in order to fit the data better. 
