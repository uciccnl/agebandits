"""

"""
import json
import numpy as np
import scipy.io
import glob

with open('subject_ids.json', 'r') as f:
    subject_ids = json.load(f)

all_mat_files = glob.glob('../transformed_Data_06.13.22_FILES/*')

subject_info = {}
for mat_file_path in all_mat_files:
    # print(mat_file_path)
    # mat_file = f'transformed_Data_06.13.22_{idx}_737960_{id}.mat'
    mat_file = mat_file_path.split('/')[-1]
    subidx = mat_file.split('.')[2].split('_')[1]
    subject_id = mat_file.split('_')[-1].split('.')[0]
    # print(subidx, subject_id)
    subject_mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    # print(subject_mat)
    subject_info[subject_id] = {
        'age': subject_mat['age'],
        'sex': subject_mat['sex'],
        'mst': subject_mat['mst'],
        'ldi': subject_mat['ldi'],
        'pss': subject_mat['pss'],
        'quic': subject_mat['quic'],
        'expDate': subject_mat['expDate'],
    }
    # print(subject_info)
    # break

with open('subject_attrsinfo.json', 'w') as f:
    json.dump(subject_info, f)

print("total subjects info", len(subject_info))
