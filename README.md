# DataCleaning
Python script for de-noising MRI data


This Python script takes pre-proccessed MRI data and performs a GLM regression on it using 36 confound parameters generated by FMRIPrep. Data is then plotted both before and after in a BOLD carpet plot for easier comparison. In order to function correctly, this script requires preprocessed MRI data generated by FMRIPrep (not included here because file sizes are very large). After data is processed, an HTML file is generated containing information and plots about and of the data. An example HTML file produced by this script is included. 

Subject parameters are entered in 'subject_script.py' and can include items like subject id's, session numbers, and task labels. This script will then automatically run each task for each subject through the data cleaning pipeline, outputting an HTML file for each subject.

This script is still a work in progress, and changes are being made constantly. 
