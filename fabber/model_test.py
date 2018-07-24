"""
Simple functions for testing model fit performance
"""

import sys
import math

import numpy as np

import nibabel as nib

from .api import Fabber, percent_progress

def _to_value_seq(values):
    """ 
    values might be a sequence or a single float value. 
    Returns either the original sequence or a sequence whose only
    item is the value
    """
    try:
        val = float(values)
        return [val,]
    except (ValueError, TypeError):
        return values

def self_test(model, options, param_testvalues, save_input=False, save_output=False, disp=True, invert=True, outfile_format="test_data_%s", **kwargs):
    """
    Run a self test on a model
    
    This consists of generating a test data set using specified parameter values, adding optional noise and then running the model
    fit on the test data
    """
    fab = Fabber(**kwargs)
    if disp: print("Running self test for model %s" % model)
    ret = {}
    options["model"] = model
    test_data = generate_test_data(fab, options, param_testvalues, param_rois=True, **kwargs)
    data, cleandata, roidata = test_data["data"], test_data["clean"], test_data["param-rois"]
    if save_input:
        outfile = outfile_format % model
        if disp: print("Saving test data to Nifti file: %s" % outfile)
        data_nii = nib.Nifti1Image(data, np.identity(4))
        data_nii.to_filename(outfile)
        if disp: print("Saving clean data to Nifti file: %s_clean" % outfile)
        cleandata_nii = nib.Nifti1Image(cleandata, np.identity(4))
        cleandata_nii.to_filename(outfile + "_clean")
        for param in param_testvalues:
            if param in roidata:
                roi_nii = nib.Nifti1Image(roidata[param], np.identity(4))
                roi_nii.to_filename(outfile + "_roi_%s" % param)
    
    log = None
    if invert:
        if disp: sys.stdout.write("Inverting test data - running Fabber:  0%%")
        sys.stdout.flush()
        if "method" not in options: options["method"] = "vb"
        if "noise" not in options: options["noise"] = "white"
        options["save-mean"] = ""
        options["save-noise-mean"] = ""
        options["save-noise-std"] = ""
        options["save-model-fit"] = ""
        options["allow-bad-voxels"] = ""
        if disp: progress_cb = percent_progress()
        else: progress_cb = None
        run = fab.run_with_data(options, {"data" : data}, progress_cb=progress_cb)
        if disp: print("\n")
        log = run.log
        if save_output:
            data_nii = nib.Nifti1Image(run.data["modelfit"], np.identity(4))
            data_nii.to_filename(outfile + "_modelfit")
        for param, values in param_testvalues.items():
            mean = run.data["mean_%s" % param]
            if save_output:
                data_nii = nib.Nifti1Image(mean, np.identity(4))
                data_nii.to_filename(outfile + "_mean_%s" % param)
            roi = roidata.get(param, np.ones(mean.shape))
            values = _to_value_seq(values)
            if len(values) > 1:
                if disp: print("Parameter: %s" % param)
                ret[param] = {}
                for idx, val in enumerate(values):
                    out = np.mean(mean[roi == idx+1])
                    if disp: print("Input %f -> %f Output" % (val, out))
                    ret[param][val] = out
        noise_mean_in = kwargs.get("noise", 0)
        noise_mean_out = np.mean(run.data["noise_means"])
        if disp: print("Noise: Input %f -> %f Output" % (noise_mean_in, 1/math.sqrt(noise_mean_out)))
        ret["noise"] = {}
        ret["noise"][noise_mean_in] = 1/math.sqrt(noise_mean_out)
        if save_output:
            data_nii = nib.Nifti1Image(run.data["noise_means"], np.identity(4))
            data_nii.to_filename(outfile + "_mean_noise")
        sys.stdout.flush()
    return ret, log

def generate_test_data(api, options, param_testvalues, nt=10, patchsize=10, noise=None, patch_rois=False, param_rois=False):
    """ 
    Generate a test Nifti image based on model evaluations

    Returns the image itself - this can be saved to a file using to_filename
    """
    dim_params = []
    dim_values = []
    dim_sizes = []
    fixed_params = {}
    for param, values in param_testvalues.items():
        values = _to_value_seq(values)
        if len(values) == 1: 
            fixed_params[param] = values[0]
        else:
            dim_params.append(param)
            dim_values.append(values)
            dim_sizes.append(len(values))

    if len(dim_sizes) > 3: 
        raise RuntimeError("Test image can only have up to 3 dimensions, you supplied %i varying parameters" % len(dim_sizes))
    else:
        for _ in range(len(dim_sizes), 3):
            dim_params.append(None)
            dim_values.append([])
            dim_sizes.append(1)

    shape = [d * patchsize for d in dim_sizes]
    data = np.zeros(shape + [nt,])
    if patch_rois: 
        patch_roi_data = np.zeros(shape)

    if param_rois:
        param_roi_data = {}
        for param in dim_params:
            if param is not None: 
                param_roi_data[param] = np.zeros(shape)

    # I bet there's a neater way to do this!
    patch_label = 1
    for x in range(dim_sizes[0]):
        for y in range(dim_sizes[1]):
            for z in range(dim_sizes[2]):
                pos = [x, y, z]
                for idx, param in enumerate(dim_params):
                    if param is not None:
                        param_value = dim_values[idx][pos[idx]]
                        fixed_params[param] = param_value
                        if param_rois:
                            param_roi_data[param][x*patchsize:(x+1)*patchsize, y*patchsize:(y+1)*patchsize, z*patchsize:(z+1)*patchsize] = pos[idx]+1
                model_curve = api.model_evaluate(options, fixed_params, nt)
                
                data[x*patchsize:(x+1)*patchsize, y*patchsize:(y+1)*patchsize, z*patchsize:(z+1)*patchsize, :] = model_curve
                if patch_rois: 
                    patch_roi_data[x*patchsize:(x+1)*patchsize, y*patchsize:(y+1)*patchsize, z*patchsize:(z+1)*patchsize] = patch_label
                    patch_label += 1

    ret = {"clean" : data}
    if noise is not None:
        # Add Gaussian noise
        #mean_signal = np.mean(data)
        noise = np.random.normal(0, noise, shape + [nt,])
        noisy_data = data + noise
        ret["data"] = noisy_data
    else:
        ret["data"] = data

    if patch_rois: ret["patch-rois"] = patch_roi_data
    if param_rois: ret["param-rois"] = param_roi_data

    return ret
