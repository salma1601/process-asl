"""
===================
Coregistration demo
===================

This example shows a basic coregistration step from anatomical to mean
functional.
"""
# Load functional ASL and anatomical images of KIRBY dataset first subject
import os
from procasl import datasets
kirby = datasets.fetch_kirby(subjects=[4])
raw_anat = kirby.anat[0]

# Create a memory context
from nipype.caching import Memory
cache_directory = '/tmp'
mem = Memory('/tmp')
os.chdir(cache_directory)

# Compute mean functional
from procasl import preprocessing
average = mem.cache(preprocessing.Average)
out_average = average(in_file=kirby.asl[0])
mean_func = out_average.outputs.mean_image

# Coregister anat to mean functional
from nipype.interfaces import spm
coregister = mem.cache(spm.Coregister)
out_coregister = coregister(
    target=mean_func,
    source=raw_anat,
    write_interp=3)

# Check coregistration
import matplotlib.pylab as plt
from nilearn import plotting
for anat, label in zip([raw_anat, out_coregister.outputs.coregistered_source],
                       ['native', 'coregistered']):
    figure = plt.figure(figsize=(5, 4))
    display = plotting.plot_anat(
        anat, figure=figure, display_mode='z', cut_coords=(-66,),
        title=label + ' anat edges on mean functional')
    display.add_edges(mean_func)
plotting.show()
