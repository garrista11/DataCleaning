# Description: Script for specifying data to be cleaned. WIP: Make more useable/
# accessible. 

from venv import create
from data_cleaning import *
import time


# Parameters for subjects, sessions, and tasks

subject_list = ['sub-01', 'sub-02']
ses = ['ses-01']
task_list1 = ['task-local_run-1', 'task_local_run-2']

# Pipeline for cleaning
for subject in subject_list:
    for task in task_list1:
        tic = time.time()
        clean_subject(subject, ses[0], task)
        toc = time.time()
        print(f"Seconds Elapsed: {toc - tic}")
    create_html(subject, task_list1, 2)
    print(f"HMTL finished for subject: {subject}")





