# 297-project-6
Project on Naive Bayes Classifier for csci 297.

## EDA
For the classification, we first label encoded the income as it was a binary classification. We dropped the categorical columns but only for the heatmap, keeping the categorical for classification. We used pd.get_dummies in order to one-hot encode all of the categorical data so that we could use them for classification. After running the heatmap, we noticed that fnlwgt has a near-zero correlation with the income and all other variables so we dropped it from the data. We kept all the other variables as they had almost no correlation with eachother but had higher correlation with the income. This is extremely advantageous for the naive bayes classifier because the naive assumption is that the fetures are entirely independent of eachother which is shown true by the heatmap. The 109-feature one-hot encoded dataframe was obviously too large to run a heatmap on so we *

## Models 
We looked at all the types of Naive Bayes classifiers and tested every variation except for the core memory overload version. There are no hyperparameters for the naive bayes models so the hyperparametrization is basically which variation of the model we select. We found the Gaussian works well because it works best with balanced data which we found that the dataset is. The multinominal and the complement performed poorly because they work better with unbalanced data which we believe this set is. The Bernoulli and categorical models worked best because they work by assigning the features either binary (Bernoulli) or categorical (categorical) values which we believe lends it better towards assigning a binary classification for income prediction. We ran confusion matrices for the two best performing models as well as classification reports to get the f1 score, precision, and recall.



