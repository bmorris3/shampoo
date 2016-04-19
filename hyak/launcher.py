from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from glob import glob
import numpy as np

python_path = '/usr/lusers/bmmorris/miniconda2/bin/python'
data_dir = '/gscratch/stf/bmmorris/shamu/2015.12.15_17-47'
output_dir = '/gscratch/stf/bmmorris/shamu/outputs'
python_script = '/usr/lusers/bmmorris/git/shampoo/hyak/hyak_jobs.py'
raw_hologram_paths = sorted(glob(os.path.join(data_dir, '*_holo.tif')))

submit_template = open('submit_template.sh', 'r').read()
walltime = '01:00:00'
email = 'bmmorris@uw.edu'

# Divide holograms to assign 14 per node at a time
n_jobs_per_node = 16
n_repeats_per_node = 2
all_hologram_indices = np.arange(len(raw_hologram_paths))
hologram_index_groups = np.array_split(all_hologram_indices,
                                       len(all_hologram_indices) //
                                       (n_jobs_per_node*n_repeats_per_node) + 1)

for i, split_hologram_indices in enumerate(hologram_index_groups):

    hologram_paths = [raw_hologram_paths[j] for j in split_hologram_indices]

    # Create input jobs to pipe to "parallel" command:
    command_list_path = os.path.join(output_dir,
                                     'command_list_{0:02d}.txt'.format(i))
    with open(command_list_path, 'w') as command_file:
        for holo_path in hologram_paths:
            line = "{0} {1} {2} {3}\n".format(python_path, python_script,
                                              holo_path, output_dir)
            command_file.write(line)

    submit_script_name = os.path.join(output_dir,
                                      'submit_script_{0:02d}.sh'.format(i))

    submit_script = submit_template.format(job_name="shampoo_test",
                                           run_dir=output_dir,
                                           log_dir=output_dir,
                                           walltime=walltime,
                                           email=email,
                                           command_list_path=command_list_path,
                                           n_jobs_per_node=n_jobs_per_node)

    submit_script_path = os.path.join(output_dir, submit_script_name)
    with open(submit_script_path, 'w') as f:
        f.write(submit_script)

    os.system('qsub {0}'.format(submit_script_path))
