"""
PYFAB - Generating and fitting test data for Fabber models
==========================================================
"""

import sys
import math

import numpy as np

import nibabel as nib

from .api_cl import FabberCl as Fabber
from .api import percent_progress

def _to_value_seq(values):
    """ 
    :param values: Either a single numeric value or a sequence of numeric values
    :return: A sequence, possibly of length one, of all the values in ``values``
    """
    try:
        val = float(values)
        return [val,]
    except (ValueError, TypeError):
        return values

def self_test(model, options, param_testvalues, save_input=False, save_output=False, disp=True, 
              invert=True, output_name="", outfile_format="test_data_%s", **kwargs):
    """
    Run a self test on a model
    
    This consists of generating a test data set using specified parameter values, adding optional 
    noise and then running the model fit on the test data

    :param model: Name of the model to test
    :param options: Model options as would be specified when fitting the model to the test data
                    to be generated
    :param param_testvalues: Mapping from parameter names (as reported by ``get_model_params``)
                             to either a single value or a sequence of values. At most 3 parameters
                             are allowed to have multiple values - this is so a 3D test volume
                             can be created where each dimension represents the variation in 
                             a single parameter.
    :param save_input: If True, save the input test data prior to running the model fitting
    :param save_output: If True, save the fitted output parameter images
    :param disp:
    :param invert: If True, run model fitting on the test data generated
    :param output_name: Name to be used in output files
    :param outfile_format: Format for output files. Should contain a single ``%s`` specifier
                           which will be replaced with ``output_name``

    Remaining keyword arguments are passed to ``generate_test_data``
    """
    fab = Fabber()
    if disp: print("Running self test for model %s" % model)
    ret = {}
    options["model"] = model
    test_data = generate_test_data(fab, options, param_testvalues, param_rois=True,
                                   output_name=output_name, **kwargs)
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
        if output_name != "":
            raise ValueError("Can't invert test data when requesting non-default output")
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
        options["data"] = data
        run = fab.run(options, progress_cb=progress_cb)
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

def generate_test_data(api, options, param_testvalues, nt=10, patchsize=10, noise=None, 
                       patch_rois=False, param_rois=False, output_name=""):
    """ 
    Generate a test data based on model evaluations for a range of parameter values

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
                model_curve = api.model_evaluate(options, fixed_params, nt, output_name=output_name)
                
                data[x*patchsize:(x+1)*patchsize, y*patchsize:(y+1)*patchsize, z*patchsize:(z+1)*patchsize, :] = model_curve
                if patch_rois: 
                    patch_roi_data[x*patchsize:(x+1)*patchsize, y*patchsize:(y+1)*patchsize, z*patchsize:(z+1)*patchsize] = patch_label
                    patch_label += 1

    ret = {"clean" : data}
    if noise is not None and noise > 0:
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
