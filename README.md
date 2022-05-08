# PET-analysis project

A full code- and datarepository for the PET-analysis projectwork. 

The data-folder contains patient-data as csv-files, with format: 
[time/minutes, Arterial blood activity, ROI1 activity, ROI2 activity, ROI3 activity, ROI4 activity, ROI5 activity]


To use the repo, please git clone the whole repo to a local directory. An outline of the files provided: 

1. **initialAssessment.py**: May be used for reading data from the data-foledr, and plotting tissue-curves, calculating mean tissue-curves, etc. for all patients.
2. **normality_test.py**: A small script used for applying the Shapiro-Wilk to the dataset. The result gave us reason for implementing Support Vector Machine-methods instead of LDA. 
3. **svm_fun.py**: A script implementing different Support Vector Machines built for predicting healthy/sick class-labels based on 6 training-patients. Uses arterial tracer activity-measurements.
4. **svm_nocvf.py**: A script implementing different Support Vector Machines built for predicting healthy/sick class-labels based on 6 training-patients. Does not use arterial tracer activity-measurements.
5. **systemMatrix.py**: Calculates P's and k's for all patients based on the data provided, using the method of least squares. 
6. **treshold_integrals.py**: Used for building a threshold-classifier based on the cumulated area under activity curves. Bar-charts are plotted. 
7. **main.py**: an object-oriented implementation of systemMatrix, initialAssessment and the svm-scripts.
8. **integrate.py**: Backbone for **main.py**. 
