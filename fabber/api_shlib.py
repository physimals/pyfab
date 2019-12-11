"""
PYFAB API using Fabber shared libraries
=======================================

This API uses the C API defined in the ``libfabbercore_shared.so`` shared library via
the Python ``ctypes`` module.
"""

import os
import collections
from ctypes import CDLL, c_int, c_char_p, c_void_p, c_uint, CFUNCTYPE, create_string_buffer

import six
import numpy as np
import numpy.ctypeslib as npct
import nibabel as nib

from .api import FabberApi, FabberException, FabberRun

class FabberShlib(FabberApi):
    """
    Interface to Fabber in library mode using simplified C API
    """

    def __init__(self, core_lib=None, model_libs=None, **kwargs):
        FabberApi.__init__(self, core_lib=core_lib, model_libs=model_libs, **kwargs)

        if self.core_lib is None or not os.path.isfile(self.core_lib):
            raise FabberException("Invalid core library - file not found: %s" % self.core_lib)

        self._errbuf = create_string_buffer(255)
        self._outbuf = create_string_buffer(1000000)
        self._progress_cb_type = CFUNCTYPE(None, c_int, c_int)
        self._clib = self._init_clib()
        self._handle = None
        self._init_handle()

    def get_methods(self):
        self._init_handle()
        self._trycall(self._clib.fabber_get_methods, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        return self._outbuf.value.decode("UTF-8").splitlines()

    def get_models(self, model_group=None):
        self._init_handle(model_group)
        self._trycall(self._clib.fabber_get_models, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        return self._outbuf.value.decode("UTF-8").splitlines()

    def get_options(self, generic=None, method=None, model=None):
        self._init_handle()
        if generic is None:
            # For backwards compatibility - no params = generic
            generic = not method and not model

        ret, all_lines = [], []
        if method:
            self._trycall(self._clib.fabber_get_options, self._handle, "method", method, len(self._outbuf), self._outbuf, self._errbuf)
            lines = self._outbuf.value.decode("UTF-8").split("\n")
            ret.append(lines[0])
            all_lines += lines[1:]
        if model:
            self._trycall(self._clib.fabber_get_options, self._handle, "model", model, len(self._outbuf), self._outbuf, self._errbuf)
            lines = self._outbuf.value.decode("UTF-8").split("\n")
            ret.append(lines[0])
            all_lines += lines[1:]
        if generic:
            self._trycall(self._clib.fabber_get_options, self._handle, None, None, len(self._outbuf), self._outbuf, self._errbuf)
            lines = self._outbuf.value.decode("UTF-8").split("\n")
            ret.append(lines[0])
            all_lines += lines[1:]
        
        opt_keys = ["name", "description", "type", "optional", "default"]
        opts = []
        for opt in all_lines:
            if opt:
                opt = dict(zip(opt_keys, opt.split("\t")))
                opt["optional"] = opt["optional"] == "1"
                opts.append(opt)
        ret.insert(0, opts)
        return tuple(ret)

    def get_model_params(self, options):
        return self._init_run(options)[1]
        
    def get_model_param_descs(self, options):
        self._init_run(options)
        # Set the arg types here because we do not know if this method will actually exist in the
        # shared library but if the user calls this method we assume they know it does.
        self._clib.fabber_get_model_param_descs.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
        self._trycall(self._clib.fabber_get_model_param_descs, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        return self._parse_params(self._outbuf.value.decode("UTF-8").splitlines())

    def get_model_outputs(self, options):
        return self._init_run(options)[2]

    def model_evaluate(self, options, param_values, nvols, indata=None, output_name=""):
        # Get model parameter names and form a sequence of the values provided for them
        _, params, _ = self._init_run(options)

        plist = []
        for param in params:
            if param not in param_values:
                raise FabberException("Model parameter %s not specified" % param)
            else:
                plist.append(param_values[param])

        if len(param_values) != len(params):
            raise FabberException("Incorrect number of parameters specified: expected %i (%s)" % (len(params), ",".join(params)))

        ret = np.zeros([nvols,], dtype=np.float32)
        if indata is None: 
            indata = np.zeros([nvols,], dtype=np.float32)

        # Call the evaluate function in the C API
        self._trycall(self._clib.fabber_model_evaluate_output, self._handle, len(plist), np.array(plist, dtype=np.float32), nvols, indata, output_name, ret, self._errbuf)

        return ret

    def run(self, options, progress_cb=None):
        if not "data" in options:
            raise ValueError("Main voxel data not provided")

        # Initialize the run, set the options and return the model parameters
        shape, params, extra_outputs = self._init_run(options)
        nvoxels = shape[0] * shape[1] * shape[2]

        output_items = []
        if "save-mean" in options:
            output_items += ["mean_" + p for p in params]
        if "save-std" in options:
            output_items += ["std_" + p for p in params]
        if "save-zstat" in options:
            output_items += ["zstat_" + p for p in params]
        if "save-noise-mean" in options:
            output_items.append("noise_means")
        if "save-noise-std" in options:
            output_items.append("noise_stdevs")
        if "save-free-energy" in options:
            output_items.append("freeEnergy")
        if "save-model-fit" in options:
            output_items.append("modelfit")
        if "save-residuals" in options:
            output_items.append("residuals")
        if "save-mvn" in options:
            output_items.append("finalMVN")
        if "save-model-extras" in options:
            output_items += extra_outputs

        progress_cb_func = self._progress_cb_type(0)
        if progress_cb is not None:
            progress_cb_func = self._progress_cb_type(progress_cb)

        retdata, log = {}, ""
        self._trycall(self._clib.fabber_dorun, self._handle, len(self._outbuf), self._outbuf, self._errbuf, progress_cb_func)
        log = self._outbuf.value.decode("UTF-8")
        for key in output_items:
            size = self._trycall(self._clib.fabber_get_data_size, self._handle, key, self._errbuf)
            arr = np.ascontiguousarray(np.empty(nvoxels * size, dtype=np.float32))
            self._trycall(self._clib.fabber_get_data, self._handle, key, arr, self._errbuf)
            if size > 1:
                arr = arr.reshape([shape[0], shape[1], shape[2], size], order='F')
            else:
                arr = arr.reshape([shape[0], shape[1], shape[2]], order='F')
            retdata[key] = arr

        return FabberRun(retdata, log)
        
    def _init_run(self, options):
        options = dict(options)
        self._init_handle()
        shape = self._set_options(options)
        self._trycall(self._clib.fabber_get_model_params, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        params = self._outbuf.value.decode("UTF-8").splitlines()
        self._trycall(self._clib.fabber_get_model_outputs, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        extra_outputs = self._outbuf.value.decode("UTF-8").splitlines()
        return shape, params, extra_outputs

    def _init_handle(self, model_group=None):
        # This is required because currently there is no CAPI function to clear the options.
        # So we destroy the old Fabber handle and create a new one
        self._destroy_handle()    
        self._handle = self._clib.fabber_new(self._errbuf)
        if self._handle is None:
            raise RuntimeError("Error creating fabber context (%s)" % self._errbuf.value.decode("UTF-8"))

        if model_group is not None:
            model_group = model_group.lower()
            if model_group not in self.model_libs:
                raise ValueError("Unknown model library: %s" % model_group)
            self._trycall(self._clib.fabber_load_models, self._handle, self.model_libs[model_group], self._errbuf)
        else:
            # Load all model libraries
            for lib in self.model_libs.values():
                self._trycall(self._clib.fabber_load_models, self._handle, lib, self._errbuf)

    def _set_options(self, options):
        # Separate out data options from 'normal' options
        data_options = {}
        model_options = self.get_options(model=options.get("model", "poly"))[0]
        for key in list(options.keys()):
            if self.is_data_option(key, model_options):
                # Allow input data to be given as Numpy array, Nifti image or filename
                value = options.pop(key)
                if value is None:
                    pass
                elif isinstance(value, nib.Nifti1Image):
                    data_options[key] = value.get_data()
                elif isinstance(value, six.string_types):
                    data_options[key] = nib.load(value).get_data()
                elif isinstance(value, np.ndarray):
                    data_options[key] = value
                else:
                    raise ValueError("Unsupported type for input data: %s = %s" % (key, type(value)))
            elif self.is_matrix_option(key, model_options):
                # Input matrices can be given as Numpy arrays or sequences but must be
                # passed to fabber as file names
                value = options.get(key)
                if isinstance(value, six.string_types):
                    pass
                elif isinstance(value, np.ndarray):
                    options[key] = self._write_temp_matrix(value)
                elif isinstance(value, collections.Sequence):
                    options[key] = self._write_temp_matrix(value)
                else:
                    raise ValueError("Unsupported type for input matrix: %s = %s" % (key, type(value)))

        # Set 'normal' options (i.e. not data items)
        for key, value in options.items():
            # Options with 'None' values are ignored
            if value is None: continue
                
            # Key separators can be specified as underscores or hyphens as hyphens are not allowed in Python
            # keywords. They are always passed as hyphens except for the anomolous PSP_byname options
            if not key.startswith("PSP_"):
                key = key.replace("_", "-")

            # Fabber interprets boolean values as 'option given=True, not given=False'. For options with the
            # value True, the actual option value passed must be blank
            if isinstance(value, bool):
                if value:
                    value = ""
                else:
                    continue
            self._trycall(self._clib.fabber_set_opt, self._handle, str(key), str(value), self._errbuf)

        # Shape comes from the main data, or if not present (e.g. during model_evaluate), take
        # shape from any data item or as single-voxel volume
        if "data" in data_options:
            shape = data_options["data"].shape
        elif data_options:
            shape = data_options[data_options.keys()[0]].shape
        else:
            shape = (1, 1, 1)
        nvoxels = shape[0] * shape[1] * shape[2]

        # Make mask suitable for passing to int* c function
        mask = data_options.pop("mask", np.ones(nvoxels))
        mask = np.ascontiguousarray(mask.flatten(order='F'), dtype=np.int32)

        # Set data options
        self._trycall(self._clib.fabber_set_extent, self._handle, shape[0], shape[1], shape[2], mask, self._errbuf)
        for key, item in data_options.items():
            if len(item.shape) == 3:
                size = 1
            else:
                size = item.shape[3]
            item = np.ascontiguousarray(item.flatten(order='F'), dtype=np.float32)
            self._trycall(self._clib.fabber_set_data, self._handle, key, size, item, self._errbuf)
        
        return shape

    def _destroy_handle(self):
        if hasattr(self, "_handle"):
            if self._handle:
                self._clib.fabber_destroy(self._handle)
                self._handle = None

    def __del__(self):
        self._destroy_handle()

    def _init_clib(self):
        try:
            clib = CDLL(str(self.core_lib))

            # Signatures of the C functions
            c_int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
            c_float_arr = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

            clib.fabber_new.argtypes = [c_char_p]
            clib.fabber_new.restype = c_void_p
            clib.fabber_load_models.argtypes = [c_void_p, c_char_p, c_char_p]
            clib.fabber_set_extent.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_int_arr, c_char_p]
            clib.fabber_set_opt.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
            clib.fabber_set_data.argtypes = [c_void_p, c_char_p, c_uint, c_float_arr, c_char_p]
            clib.fabber_get_data_size.argtypes = [c_void_p, c_char_p, c_char_p]
            clib.fabber_get_data.argtypes = [c_void_p, c_char_p, c_float_arr, c_char_p]
            clib.fabber_dorun.argtypes = [c_void_p, c_uint, c_char_p, c_char_p, self._progress_cb_type]
            clib.fabber_destroy.argtypes = [c_void_p]

            clib.fabber_get_options.argtypes = [c_void_p, c_char_p, c_char_p, c_uint, c_char_p, c_char_p]
            clib.fabber_get_models.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            clib.fabber_get_methods.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]

            clib.fabber_get_model_params.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            clib.fabber_get_model_outputs.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            clib.fabber_model_evaluate.argtypes = [c_void_p, c_uint, c_float_arr, c_uint, c_float_arr, c_float_arr, c_char_p]
            clib.fabber_model_evaluate_output.argtypes = [c_void_p, c_uint, c_float_arr, c_uint, c_float_arr, c_char_p, c_float_arr, c_char_p]
            return clib
        except Exception as exc:
            raise RuntimeError("Error initializing Fabber library: %s" % str(exc))

    def _trycall(self, call, *args):
        # Need to pass strings as byte-strings - assume UTF-8
        # although nothing in Fabber goes beyond ASCII right now
        new_args = []
        for arg in args:
            if isinstance(arg, six.string_types):
                new_args.append(arg.encode("UTF-8"))
            else:
                new_args.append(arg)
        ret = call(*new_args)
        if ret < 0:
            raise FabberException(self._errbuf.value.decode("UTF-8"), ret, self._outbuf.value.decode("UTF-8"))
        else:
            return ret
