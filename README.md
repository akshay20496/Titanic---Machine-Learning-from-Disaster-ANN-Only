# Titanic---Machine-Learning-from-Disaster

**The Challenge**
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, they ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” 

**train.csv** will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

**test.csv** dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

**Evaluation**
**Goal**
It is our job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, we must predict a 0 or 1 value for the variable.

**Metric**
Our score is the percentage of passengers you correctly predict. This is known as accuracy.

**Submission File Format**
We should submit a csv file with exactly 418 entries plus a header row. Our submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.
Submit a csv file with exactly 418 entries plus a header row. 
The file should have exactly 2 columns:
          - PassengerId (sorted in any order)
          - Survived (contains your binary predictions: 1 for survived, 0 for deceased)


**Citation**
Will Cukierski. Titanic - Machine Learning from Disaster. https://kaggle.com/competitions/titanic, 2012. Kaggle.

