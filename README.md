python_ML

#20171022 Knowledge is a treasure but practice is the key to it
chapter1\p030LogisticRegression_ComprehensievePra.py <br>
LogisticRegression:  from sklearn.linear_mode <br>
# sigmoid(x*coef[0] + y*coef[1] + intercept) = type
data dimension: 2 dimensions -- Cell Size & Clump Thickness <br>
distribution: ?? <br>
source: cvs <br>
pre: read by pandas.read_csv; separate according to 'type' and extract field 'Cell Size','Clump Thickness' by read_data.loc <br>
scenarios: binary classification <br>

(2.1.1.3)Naive Bayes Model for News Text Classification chapter2 <br>
# naive bayes model
MultinomialNB: from sklearn.naive_bayes
# formulation:
data dimension:??  text <br>
distribution: <br>
source: from sklearn.datasets import fetch_20newsgroups(data_set) parameter: subset=train,test,all <br>
pre: sklearn.cross_validation.train_test_split into X,Y and tran,test ; (Y is 0-20, the number of each class vary from 470-766) <br>
sklearn.feature_extraction.text.CountVectorizer, train the vocabulary dictionary into tern document matrix <br>
scenarios: text extract and classification <br>
report: sklearn.metrics.classification_report : about precision, recall, f1-score, support <br>




# 201710
# I think we need to annotate the data features(like dimension / linearity / distribution / source / scenarios),
# the algorithm we use, and why it is suitable to the data.



# 201709
## new word: coefficient  intercept 
##  lib name: matplotlib.pyplot -> plt
       sklearn.linear_model


# Github
1\create a new repository "python_ML", on the website
2\on the command line in your current project folder, enter: 
# create a file README.md
echo "# python_ML" >> README.md  
# initiate git in current
git init
# add new file want to upload
git add README.md
# commit and give some information
git commit -m "first commit"
# this is license file
git remote add origin https://github.com/TaoStarlit/python_ML.git

# push to remote need your account and password
git push -u origin master
# done