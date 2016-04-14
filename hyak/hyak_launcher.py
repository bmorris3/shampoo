from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from glob import glob

python_path = '/usr/lusers/bmmorris/miniconda2/bin/python'
data_dir = '/gscratch/stf/bmmorris/shamu/2015.12.15_17-47'
outputs_dir = '/gscratch/stf/bmmorris/shamu/outputs'
python_script = '/usr/lusers/bmmorris/git/shampoo/hyak/hyak_jobs.py'
raw_hologram_paths = sorted(glob(os.path.join(data_dir, '*_holo.tif')))

# Create input jobs to "parallel" command:

with open('command_list.txt', 'w') as command_file:
    for holo_path in raw_hologram_paths:
        line = "{0} {1} {2} {3}\n".format(python_path, python_script, holo_path,
                                          output_dir)
        command_file.write(line)