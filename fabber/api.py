"""
Python API for the the FSL Fabber tool using the C API via the Python ctypes library
"""

import os
import sys
import warnings
import datetime
import time
import glob
import collections
import tempfile
from ctypes import CDLL, c_int, c_char_p, c_void_p, c_uint, CFUNCTYPE, create_string_buffer

import six
import numpy as np
import numpy.ctypeslib as npct
import nibabel as nib

if sys.platform.startswith("win"):
    _LIB_FORMAT = "bin\\%s.dll"
    _BIN_FORMAT = "bin\\%s.exe"
elif sys.platform.startswith("darwin"):
    _LIB_FORMAT = "lib/lib%s.dylib"
    _BIN_FORMAT = "bin/%s"
else:
    _LIB_FORMAT = "lib/lib%s.so"
    _BIN_FORMAT = "bin/%s"

def percent_progress(log=sys.stdout):
    """
    :return: Convenience progress callback which updates a percentage on the specified output stream
    """
    def _progress(voxel, nvoxels):
        complete = 100*voxel/nvoxels
        log.write("\b\b\b\b%3i%%" % complete)
        log.flush()
    return _progress

def _find_file(current_value, envdir, search_for):
    if current_value is not None:
        return current_value
    elif envdir in os.environ:
        newfpath = os.path.join(os.environ[envdir], search_for)
        if os.path.isfile(newfpath):
            return newfpath
        else:
            return current_value
    else:
        return None

def find_fabber():
    """
    Find the Fabber executable, core library and model libraries, or return None if not found

    :return: A tuple of executable, core library, sequence of model libraries
    """
    ex, lib, models = None, None, []
    for envdir in ("FABBERDIR", "FSLDEVDIR", "FSLDIR"):
        ex = _find_file(ex, envdir, _BIN_FORMAT % "fabber")
        lib = _find_file(lib, envdir, _LIB_FORMAT % "fabbercore_shared")
        models += glob.glob(os.path.join(os.environ.get(envdir, ""), _LIB_FORMAT % "fabber_models_*"))

    return ex, lib, models

def load_options_files(fname):
    """ 
    Load options for a Fabber run from an .fab options file

    :param fname: File name of options file
    """
    options = {}
    with open(fname, "r") as fabfile:
        for line in fabfile.readlines():
            line = line.strip()
            if line and line[0] != "#":
                keyval = line.split("=", 1)
                key = keyval[0].strip()
                if len(keyval) > 1:
                    value = keyval[1].strip()
                else:
                    value = True
                options[key] = value

    return options

def save_options_file(options, fname):
    """
    Save options as a .fab file.
    """
    with open(fname, "w") as fabfile:
        dump_options_file(options, fabfile)
        
def dump_options_file(options, stream):
    """
    Dump to an output stream

    :param stream: Output stream (e.g. stdout or fileobj)
    """
    for key in sorted(options.keys()):
        value = options[key]
        if value == "" or (isinstance(value, bool) and value):
            stream.write("%s" % key)
        elif not isinstance(value, bool):
            stream.write("%s=%s" % (key, value))
        stream.write("\n")

class FabberException(RuntimeError):
    """
    Thrown if there is an error using the Fabber executable or library
    """
    def __init__(self, msg, errcode=None, log=None):
        self.errcode = errcode
        self.log = log
        if errcode is not None:
            RuntimeError.__init__(self, "%i: %s" % (errcode, msg))
        else:
            RuntimeError.__init__(self, msg)

class FabberRun(object):
    """
    The result of a Fabber run
    """
    def __init__(self, data, log):
        self.data = data
        self.log = log
        self.timestamp, self.timestamp_str = self._get_log_timestamp(self.log)

    def write_to_dir(self, dirname, ref_nii=None, extension=".nii.gz"):
        """
        Write the run output to a directory

        This aspires to write the output in a form as close to the command line tool
        as possible, however exact agreement is not guaranteed
        
        :param dirname: Name of directory to write to, will be created if it does not exist
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        if not os.path.isdir(dirname):
            raise IOError("Specified directory '%s' exists but is not a directory" % dirname)

        if ref_nii:
            header = ref_nii.header
            affine = ref_nii.header.get_best_affine()
        else:
            header = None
            affine = np.identity(4)

        for data_name, arr in self.data.items():
            nii = nib.Nifti1Image(arr, header=header, affine=affine)
            nii.to_filename(os.path.join(dirname, "%s%s" % (data_name, extension)))
        
        with open(os.path.join(dirname, "logfile"), "w") as logfile:
            logfile.write(self.log)

    def _get_log_timestamp(self, log):
        prefixes = ["start time:", "fabberrundata::start time:"]
        timestamp_str = ""
        for line in log.splitlines():
            line = line.strip()
            for prefix in prefixes:
                if line.lower().startswith(prefix):
                    timestamp_str = line[len(prefix):].strip()
                    try:
                        timestamp = time.strptime(timestamp_str)
                        return timestamp, timestamp_str
                    except ValueError:
                        warnings.warn("Failed to parse timestamp: '%s'" % timestamp_str)
        if log != "":
            warnings.warn("Could not find timestamp in log")
        return datetime.datetime.now(), timestamp_str

class Fabber(object):
    """
    Interface to Fabber in library mode using simplified C API
    """

    BOOL = "BOOL"
    STR = "STR"
    INT = "INT",
    FLOAT = "FLOAT"
    FILE = "FILE"
    IMAGE = "IMAGE"
    TIMESERIES = "TIMESERIES"
    MVN = "MVN"
    MATRIX = "MATRIX"

    def __init__(self, core_lib=None, model_libs=None):
        self.ex, self.core_lib, self.model_libs = find_fabber()

        if core_lib:
            self.core_lib = core_lib

        if self.core_lib is None or not os.path.isfile(self.core_lib):
            raise FabberException("Invalid core library - file not found: %s" % self.core_lib)

        if model_libs is not None:
            self.model_libs = set(model_libs)

        for lib in self.model_libs:
            if not os.path.isfile(lib):
                raise FabberException("Invalid models library - file not found: %s" % lib)
           
        self._errbuf = create_string_buffer(255)
        self._outbuf = create_string_buffer(1000000)
        self._progress_cb_type = CFUNCTYPE(None, c_int, c_int)
        self._clib = self._init_clib()
        self._handle = None
        self._init_handle()

    def get_methods(self):
        """ 
        Get known inference methods
        
        :return: Sequence of known inference method names
        """
        self._trycall(self._clib.fabber_get_methods, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        return self._outbuf.value.decode("UTF-8").splitlines()

    def get_models(self):
        """ 
        Get known models
        
        :return: Sequence of known model names
        """
        self._trycall(self._clib.fabber_get_models, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        return self._outbuf.value.decode("UTF-8").splitlines()

    def get_options(self, generic=None, method=None, model=None):
        """
        Get known Fabber options

        :param method: If specified, return options for this method
        :param model: If specified, return options for this model
        :param generic: If True, return generic Fabber options

        If no parameters are specified, generic options only are returned

        :return: Tuple of options, method description, model_description, generic_description. 
                 Descriptions are only included if relevant options were requestsed. Options 
                 is a list of options, each in the form of a dictionary.
                 Descriptions are simple text descriptions of the method or model
        """
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
        """ 
        Get model parameters
        
        :param options: Options dictionary
        :return: Sequence of model parameter names
        """
        return self._init_run(options)[1]
        
    def get_model_outputs(self, options):
        """ 
        Get additional model timeseries outputs
        
        :param options: Fabber options
        :return: Sequence of names of additional model timeseries outputs
        """
        return self._init_run(options)[2]

    def model_evaluate(self, options, param_values, nvols, indata=None):
        """
        Evaluate the model with a specified set of parameters

        :param options: Fabber options as key/value dictionary
        :param param_values: Parameter values as a dictionary of param name : param value
        :param nvols: Length of output array - equivalent to number of volumes in input data set
        """
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
        self._trycall(self._clib.fabber_model_evaluate, self._handle, len(plist), np.array(plist, dtype=np.float32), nvols, indata, ret, self._errbuf)

        return ret

    def run(self, options, progress_cb=None):
        """
        Run fabber

        :param options: Fabber options as key/value dictionary. Data may be passed as Numpy arrays, Nifti 
                        images or strings (which are interpreted as filenames)
        :param progress_cb: Callable which will be called periodically during processing

        :return: On success, a FabberRun instance
        """
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

    def _write_temp_matrix(self, matrix):
        with tempfile.NamedTemporaryFile(prefix="fab", delete=False) as tempf:
            for row in matrix:
                if isinstance(row, collections.Sequence):
                    tempf.write(" ".join(["%f" % val for val in row]) + "\n")
                else:
                    tempf.write("%f\n" % row)
            return tempf.name
        
    def is_data_option(self, key, options):
        """
        :param key: Option name
        :param options: Options as key/value dict

        :return: True if ``key`` is a voxel data option
        """ 
        if key in ("data", "mask", "suppdata", "continue-from-mvn"):
            return True
        elif key.startswith("PSP_byname") and key.endswith("_image"):
            return True
        else:
            return key in [option["name"] for option in options if option["type"] in (self.IMAGE, self.TIMESERIES, self.MVN)]

    def _is_matrix_option(self, key, model_options):
        return key in [option["name"] for option in model_options if option["type"] == self.MATRIX]

    def _init_run(self, options):
        options = dict(options)
        self._init_handle()
        shape = self._set_options(options)
        self._trycall(self._clib.fabber_get_model_params, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        params = self._outbuf.value.decode("UTF-8").splitlines()
        self._trycall(self._clib.fabber_get_model_outputs, self._handle, len(self._outbuf), self._outbuf, self._errbuf)
        extra_outputs = self._outbuf.value.decode("UTF-8").splitlines()
        return shape, params, extra_outputs

    def _init_handle(self):
        # This is required because currently there is no CAPI function to clear the options.
        # So we destroy the old Fabber handle and create a new one
        self._destroy_handle()    
        self._handle = self._clib.fabber_new(self._errbuf)
        if self._handle is None:
            raise RuntimeError("Error creating fabber context (%s)" % self._errbuf.value.decode("UTF-8"))

        for lib in self.model_libs:
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
            elif self._is_matrix_option(key, model_options):
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
