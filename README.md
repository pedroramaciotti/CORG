# CORG
### Corporal Geometrics: a module for dimensionality analysis of text corpora

Check the quickstart jupyter notebook quickstart tutorial.

CORG stands for "COrporal Geometrics". As suggested by its name, it's a module for performing inference in copora that have been embedded in some geometrical space.

Inputs are documents, binary categories for them, and positions for documents.

CORG does two things:

1) I tests whether a given space direction servers as a good classifier for some binary category of documents. For this, it fits a logistic regression model on the dimension and repots accuracy metrics.

2) Given a binary classification of documents, it finds the space direction on which we can fit the best logistic regression binary classifier, i.e., the direction that best dichotomizes the documents labeled as belonging to one of two categories.

Check our publications for further details:

Pedro Ramaciotti Morales. "Multidimensional online American politics: Mining emergent social cleavages in social graphs" (2022). The 2022 Conference on Complex Networks and their Applications.
https://hal.archives-ouvertes.fr/hal-03721438/document

### Installation

    pip install corg

### Acknowledgements

This software package been funded by the “European Polarisation Observatory” (EPO) of the CIVICA Consortium, and by Data Intelligence Institute of Paris through the French National Agency for Research (ANR) grant ANR-18-IDEX-0001 “IdEx Universite de Paris”. Data declared the 19 March 2020 and 15 July 2021 at the registry of data processing at the Fondation Nationale de Sciences Politiques (Sciences Po) in accordance with General Data Protection Reg- ulation 2016/679 (GDPR) and Twitter policy. For further details and the respec- tive legal notice, please visit https://medialab.sciencespo.fr/en/activities/epo/.
