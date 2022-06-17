# CORG
### Corporal Geometrics: a module for dimensionality analysis of text corpora

Check the quickstart jupyter notebook quickstart tutorial.

CORG stands for "COrporal Geometrics". As suggested by its name, it's a module for performing inference in copora that have been embedded in some geometrical space.

Inputs are documents, binary categories for them, and positions for documents.

CORG does two things:

1) I tests whether a given space direction servers as a good classifier for some binary category of documents. For this, it fits a logistic regression model on the dimension and repots accuracy metrics.

2) Given a binary classification of documents, it finds the space direction on which we can fit the best logistic regression binary classifier, i.e., the direction that best dichotomizes the documents labeled as belonging to one of two categories.
