# Author: Taylor :)
# Description: Script for (hopefully) cleaning MR data and comparing
# outputs using various de-noising parameters.

# Need to segment data before filtering
import os
from pkgutil import get_data
import numpy as np
import nibabel as nib
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class CleanData:
    """Creates a 'CleanData' object that will clean and organize MR data"""
    
    def __init__(self, subject_id, session, task):
        """Initializes a 'CleanData' object that takes three parameters: a subject id,
        a session number, and a task number. Parameters much match a file path."""

        # Local path - replace if different
        self._local_main = os.path.join(".", "SampleData")
        
        # Parameters
        self._subject_id = subject_id
        self._session = session
        self._task = task
        
        # Storage for necessary data
        self._o_dataset = None
        self.o_data_zscore = None
        self._filtered_datasets = []
        self._time_points = None
        self._design_matricies = []
        self._organized_datasets = []
        self._organized_locs = []
        self._b_mask = None

        # Storage for segmentation values
        self._wm_voxels = 0
        self._cortex_voxels = 0
        self._cerebellum_voxels = 0
        self._edge_voxels = 0
        self._betas = None

    def get_data(self):
        """Unpack and unwrap raw data and apply brain mask to get relevant data"""

        data_path = os.path.join(self._local_main, f"{self._subject_id}", f"{self._session}", "func", f"{self._subject_id}_{self._session}_{self._task}_space-T1w_desc-preproc_bold.nii.gz")
        raw_data = nib.load(data_path)
        raw_data = raw_data.get_fdata()
        raw_data = raw_data.reshape(-1, raw_data.shape[3])

        brain_mask_path = os.path.join(self._local_main, f"{self._subject_id}", f"{self._session}", "func", f"{self._subject_id}_{self._session}_{self._task}_space-T1w_desc-brain_mask.nii.gz")
        b_mask = nib.load(brain_mask_path)
        b_mask = b_mask.get_fdata()
        b_mask = np.array(b_mask, dtype=bool)
        b_mask = b_mask.reshape(-1,1)
        self._b_mask = b_mask.reshape(-1,1)

        self._time_points = raw_data.shape[1]

        m_slice = raw_data[:, 0].reshape(-1, 1)

        masked_data = m_slice[b_mask].reshape(-1, 1)

        for i in range(1, self._time_points):
            m_slice = raw_data[:, i].reshape(-1,1)
            m_slice = m_slice[b_mask].reshape(-1,1)
            masked_data = np.hstack((masked_data, m_slice))
        
        self._o_dataset = masked_data
        self.o_data_zscore = stats.zscore(self._o_dataset, 1)


    def get_confounds(self):
        """Get confound values from csv file and create design matrix
        WIP: Be able to change parameter list, currently set to 36p de-noising scheme"""

        confounds_path = os.path.join(self._local_main, f"{self._subject_id}", f"{self._session}", "func", f"{self._subject_id}_{self._session}_{self._task}_desc-confounds_timeseries.tsv")
        confounds = pd.read_csv(confounds_path, sep='\t')
        i_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
               'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2' ,'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
               'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
               'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2',
               'csf', 'csf_derivative1', 'csf_derivative1_power2', 'csf_power2',
               'white_matter', 'white_matter_derivative1', 'white_matter_derivative1_power2', 'white_matter_power2',
               'global_signal', 'global_signal_derivative1', 'global_signal_derivative1_power2', 'global_signal_power2']
        design_matrix = np.linspace(0,1,self._time_points).reshape(-1,1)

        # Add confounds to design matrix
        for con in i_confounds:
            np_confounds = (confounds[con]).to_numpy(na_value = 0)
            reshaped_np = np_confounds.reshape(-1,1)
            # Combine into a single array
            design_matrix = np.hstack((design_matrix, reshaped_np))
        
        self._design_matricies.append(design_matrix)

    def glm_regression(self):
        """Perform GLM regression on data
        WIP: Perform GLM regression with different sets of parameters"""
        for matrix in self._design_matricies:
            model = LinearRegression().fit(matrix, self._o_dataset.T)
            self._betas = model.coef_

            y_hat = model.predict(matrix)
            filt_y = np.subtract(self._o_dataset.T, y_hat)
            self._filtered_datasets.append(filt_y)

    def segmentation_orig(self):
        """Segment and organize the original, unfiltered data"""

        seg_path = os.path.join(self._local_main, f"{self._subject_id}", f"{self._session}", "func", f"{self._subject_id}_{self._session}_{self._task}_space-T1w_desc-aseg_dseg.nii.gz")
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata().reshape(-1,1)
        seg_data = seg_data[self._b_mask].reshape(-1,1)

        seg_wm = (seg_data == 2) | (seg_data == 41)  | (seg_data == 24)
        filt_slice = self._o_dataset[:, 0].reshape(-1,1)
        self._o_dataset_wm = filt_slice[seg_wm].reshape(-1,1)
        self._wm_voxels = self._o_dataset_wm.shape[0]

        seg_cortex = (seg_data == 3) | (seg_data == 42) 
        filt_slice = self._o_dataset[:, 0].reshape(-1,1)
        self._o_dataset_cortex = filt_slice[seg_cortex].reshape(-1,1)
        self._cortex_voxels = self._o_dataset_cortex.shape[0]

        seg_cerebellum = (seg_data == 7) | (seg_data == 8) | (seg_data == 46) | (seg_data == 47)
        filt_slice = self._o_dataset[:, 0].reshape(-1,1)
        self._o_dataset_cerebellum = filt_slice[seg_cerebellum].reshape(-1,1)
        self._cerebellum_voxels = self._o_dataset_cerebellum.shape[0]

        seg_edge = (seg_data != 2) & (seg_data != 41)  & (seg_data != 24) & (seg_data != 3) & (seg_data != 42) & (seg_data != 7) & (seg_data != 8) & (seg_data != 46) & (seg_data != 47)
        filt_slice = self._o_dataset[:, 0].reshape(-1,1)
        self._o_dataset_edge = filt_slice[seg_edge].reshape(-1,1)
        self._edge_voxels = self._o_dataset_edge.shape[0]


        sub_org_data = np.vstack((self._o_dataset_wm, self._o_dataset_cortex))
        sub_org_data = np.vstack((sub_org_data, self._o_dataset_cerebellum))
        sub_org_data = np.vstack((sub_org_data, self._o_dataset_edge))
        org_data = sub_org_data

        for n in range(1 , self._time_points):
            seg_wm = (seg_data == 2) | (seg_data == 41)  | (seg_data == 24)
            filt_slice = self._o_dataset[:, n].reshape(-1,1)
            self._o_dataset_wm = filt_slice[seg_wm].reshape(-1,1)

            seg_cortex = (seg_data == 3) | (seg_data == 42) 
            filt_slice = self._o_dataset[:, n].reshape(-1,1)
            self._o_dataset_cortex = filt_slice[seg_cortex].reshape(-1,1)

            seg_cerebellum = (seg_data == 7) | (seg_data == 8) | (seg_data == 46) | (seg_data == 47)
            filt_slice = self._o_dataset[:, n].reshape(-1,1)
            self._o_dataset_cerebellum = filt_slice[seg_cerebellum].reshape(-1,1)

            seg_edge = (seg_data != 2) & (seg_data != 41)  & (seg_data != 24) & (seg_data != 3) & (seg_data != 42) & (seg_data != 7) & (seg_data != 8) & (seg_data != 46) & (seg_data != 47)
            filt_slice = self._o_dataset[:, n].reshape(-1,1)
            self._o_dataset_edge = filt_slice[seg_edge].reshape(-1,1)

            sub_org_data = np.vstack((self._o_dataset_wm, self._o_dataset_cortex))
            sub_org_data = np.vstack((sub_org_data, self._o_dataset_cerebellum))
            sub_org_data = np.vstack((sub_org_data, self._o_dataset_edge))
            org_data = np.hstack((org_data, sub_org_data))
        self._o_dataset = org_data

    def segmentation_filt(self):
        """Segment and organize the filtered data"""

        seg_path = os.path.join(self._local_main, f"{self._subject_id}", f"{self._session}", "func", f"{self._subject_id}_{self._session}_{self._task}_space-T1w_desc-aseg_dseg.nii.gz")
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata().reshape(-1,1)
        seg_data = seg_data[self._b_mask].reshape(-1,1)
        for filt_y in self._filtered_datasets:
            seg_wm = (seg_data == 2) | (seg_data == 41)  | (seg_data == 24)
            filt_slice = filt_y.T[:, 0].reshape(-1,1)
            filt_y_wm = filt_slice[seg_wm].reshape(-1,1)
            self._wm_voxels = filt_y_wm.shape[0]

            seg_cortex = (seg_data == 3) | (seg_data == 42) 
            filt_slice = filt_y.T[:, 0].reshape(-1,1)
            filt_y_cortex = filt_slice[seg_cortex].reshape(-1,1)
            self._cortex_voxels = filt_y_cortex.shape[0]

            seg_cerebellum = (seg_data == 7) | (seg_data == 8) | (seg_data == 46) | (seg_data == 47)
            filt_slice = filt_y.T[:, 0].reshape(-1,1)
            filt_y_cerebellum = filt_slice[seg_cerebellum].reshape(-1,1)
            self._cerebellum_voxels = filt_y_cerebellum.shape[0]

            seg_edge = (seg_data != 2) & (seg_data != 41)  & (seg_data != 24) & (seg_data != 3) & (seg_data != 42) & (seg_data != 7) & (seg_data != 8) & (seg_data != 46) & (seg_data != 47)
            filt_slice = filt_y.T[:, 0].reshape(-1,1)
            filt_y_edge = filt_slice[seg_edge].reshape(-1,1)
            self._edge_voxels = filt_y_edge.shape[0]


            sub_org_data = np.vstack((filt_y_wm, filt_y_cortex))
            sub_org_data = np.vstack((sub_org_data, filt_y_cerebellum))
            sub_org_data = np.vstack((sub_org_data, filt_y_edge))
            org_data = sub_org_data

            for n in range(1 , self._time_points):
                seg_wm = (seg_data == 2) | (seg_data == 41)  | (seg_data == 24)
                filt_slice = filt_y.T[:, n].reshape(-1,1)
                filt_y_wm = filt_slice[seg_wm].reshape(-1,1)

                seg_cortex = (seg_data == 3) | (seg_data == 42) 
                filt_slice = filt_y.T[:, n].reshape(-1,1)
                filt_y_cortex = filt_slice[seg_cortex].reshape(-1,1)

                seg_cerebellum = (seg_data == 7) | (seg_data == 8) | (seg_data == 46) | (seg_data == 47)
                filt_slice = filt_y.T[:, n].reshape(-1,1)
                filt_y_cerebellum = filt_slice[seg_cerebellum].reshape(-1,1)

                seg_edge = (seg_data != 2) & (seg_data != 41)  & (seg_data != 24) & (seg_data != 3) & (seg_data != 42) & (seg_data != 7) & (seg_data != 8) & (seg_data != 46) & (seg_data != 47)
                filt_slice = filt_y.T[:, n].reshape(-1,1)
                filt_y_edge = filt_slice[seg_edge].reshape(-1,1)

                sub_org_data = np.vstack((filt_y_wm, filt_y_cortex))
                sub_org_data = np.vstack((sub_org_data, filt_y_cerebellum))
                sub_org_data = np.vstack((sub_org_data, filt_y_edge))
                org_data = np.hstack((org_data, sub_org_data))

            self._organized_datasets.append(org_data)

    def plot_and_save(self):
        """Plot the unfiltered and filtered data
        WIP: Plot the beta weights into a table"""
        x_axis_locs = [x for x in range(0,self._time_points + 1, 32)]
        x_axis_ticks = [float(x) for x in range(0, self._time_points * 2 + 1, 64)]

        try:
            os.makedirs(f"./Figures/{self._subject_id}")
        except FileExistsError:
            pass
            

        figure_2 = plt.figure(figsize = (10, 5))
        plt.imshow(stats.zscore(self._o_dataset, 1), vmin=-1, vmax=1,interpolation='nearest', aspect='auto', cmap='gray')
        plt.yticks([0, self._wm_voxels, self._wm_voxels + self._cortex_voxels, self._wm_voxels + self._cortex_voxels + self._cerebellum_voxels], ['White Matter', 'Cortical', 'Cerebellum', 'Edge'])
        plt.xticks(x_axis_locs, x_axis_ticks)
        plt.xlabel('time (s)')
        plt.savefig(os.path.join(".", "Figures", self._subject_id, f"o_data{self._task}.svg"), bbox_inches='tight')
        plt.close()

        figure_4 = plt.figure(figsize = (10, 5))
        plt.imshow(stats.zscore(self._organized_datasets[0], 1), vmin=-1, vmax=1,interpolation='nearest', aspect='auto', cmap='gray')
        plt.xticks(x_axis_locs, x_axis_ticks)
        plt.xlabel('time (s)')
        plt.yticks([0, self._wm_voxels, self._wm_voxels + self._cortex_voxels, self._wm_voxels + self._cortex_voxels + self._cerebellum_voxels], ['White Matter', 'Cortical', 'Cerebellum', 'Edge'])
        plt.savefig(os.path.join(".", "Figures", self._subject_id, f"fo_data{self._task}.svg"), bbox_inches='tight')
        plt.close()


def clean_subject(sub_id, ses, task):
    """Uses the CleanData class to send raw data through cleaning pipeline. Takes three parameters:
    subject id, session, and task name. Prints progress."""
    first_attempt = CleanData(sub_id, ses, task)
    first_attempt.get_data()
    first_attempt.get_confounds()
    first_attempt.segmentation_orig()
    first_attempt.glm_regression()
    first_attempt.segmentation_filt()
    first_attempt.plot_and_save()
    # first_attempt.create_beta_table()
    print(f"Task: {task} done for subject: {sub_id}")


def create_html(sub_id, task_list, n_images):
    """Uses images created during cleaning and outputs HTML file presenting data
    before and after processing. Takes two/three parameters: subject id, and a list of tasks."""

    img_paths = []
    img_type = ["o_data", "fo_data"]
    for task in task_list:
        for img in img_type:
            img_paths.append(os.path.join(".", "Figures", sub_id, f"{img}{task}.svg"))
    
    table_paths = []
    table_type = ["beta_weights"]
    for task in task_list:
        for table in table_type:
            table_paths.append(os.path.join(".", "Figures", sub_id, f"{table}{task}.svg"))

    f = open(f"{sub_id}.html", "w")

    html_template = f"""<html>
    <head>
        <title>{sub_id} Data Cleaning</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-Zenh87qX5JnK2Jl0vWa8Ck2rdkQ2Bzep5IDxbcnCeuOxjzrPF/et3URy9Bv1WTRi" crossorigin="anonymous">
    </head>
    
        <h1>{sub_id} Cleaned Data Comparisons</h1>
    
        <h2>BOLD Carpet Plots.</h2>
        <ul>
            <li>{len(task_list)} tasks</li>
            <li>First BOLD carpet plot is unfiltered data</li>
            <li>Second BOLD carpet plot is filtered with 36 parameters: 24 motion, 8 phys, 4 GSR</li>
        </ul>

        <h3>{task_list[0]} Original vs. Filtered Data</h3>
        <div>
            <img src={img_paths[0]} />
            <img src={img_paths[1]} />
        </div>
            

        <h3>{task_list[1]}</h3>
        <div>
            <img src={img_paths[2]} />
            <img src={img_paths[3]} />
        </div>
    
        


    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-OERcA2EqjJCMA+/3y+gxIOqMEjwtxJY7qPCqsdltbNJuaOe923+mo//f6V8Qbsw3" crossorigin="anonymous"></script>
    </body>
    </html>
    """
    
    # writing the code into the file
    f.write(html_template)
    print(img_paths[0])
    
    # close the file
    f.close()
