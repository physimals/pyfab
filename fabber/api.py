import os
import sys
import warnings
import datetime, time
import collections
import glob
import traceback
import math

from ctypes import *
import numpy as np
import numpy.ctypeslib as npct

import nibabel as nib

if sys.platform.startswith("win"):
    _lib_format = "bin\\%s.dll"
    _bin_format = "bin\\%s.exe"
elif sys.platform.startswith("darwin"):
    _lib_format = "lib/lib%s.dylib"
    _bin_format = "bin/%s"
else:
    _lib_format = "lib/lib%s.so"
    _bin_format = "bin/%s"

def _find_file(f, envdir, newf):
    if f is not None:
        return f
    elif envdir in os.environ:
        newfpath = os.path.join(os.environ[envdir], newf)
        if os.path.isfile(newfpath):
            return newfpath
        else:
            return f
    else:
        return None

def find_fabber():
    """
    Find the Fabber executable, core library and model libraries, or return None if not found

    Returns a tuple of exec, core lib, list of models
    """
    ex, lib, models = None, None, []
    for envdir in ("FABBERDIR", "FSLDEVDIR", "FSLDIR"):
        ex = _find_file(ex, envdir, _bin_format % "fabber")
        lib = _find_file(lib, envdir, _lib_format % "fabbercore_shared")
        models += glob.glob(os.path.join(os.environ.get(envdir, ""), _lib_format % "fabber_models_*"))

    return ex, lib, models

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

class FabberRunData(collections.MutableMapping):
    """
    Options for a Fabber run

    Behaves like a dict but supports save to/load from file,
    default values and keeps list of file lines separate
    so comments can be preserved
    """

    def __init__(self, filename=None):
        # Mapping of options as key/value pairs
        self.options = {}

        # Lines in the file in order, either a key or a comment string
        # Enables file to be recreated with comments in right place
        self.filelines = []

        if filename:
            self.load(filename)
        else:
            self._init_default()

            # self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getstate__(self):
        return (self.options, self.filelines, self.filepath)

    def __setstate__(self, state):
        self.options, self.filelines, self.filepath = state

    def __getitem__(self, key):
        return self.options[key.strip()]

    def __setitem__(self, key, value):
        key = key.strip()
        if key == "": return
        if key not in self.options:
            self.filelines.append(key)
        self.options[key] = value.strip()

    def __delitem__(self, key):
        key = key.strip()
        if key == "": return
        if key in self.options:
            del self.options[key]
            self.filelines.remove(key)

    def __iter__(self):
        return iter(self.options)

    def __len__(self):
        return len(self.options)

    def add_comment(self, comment, option=None):
        if option is None:
            idx = 0
        else:
            idx = self.filelines.index(option)

        self.filelines.insert(idx, "# %s" % str(comment))

    def set_file(self, f):
        """ Set the file. Do not parse, this is just used
            as part of 'save as' """
        self.filepath = os.path.abspath(f)
        # FIXME update views?

    def get_filedir(self):
        dir, junk = os.path.split(self.filepath)
        return dir

    def get_filename(self):
        junk, fname = os.path.split(self.filepath)
        return fname

    def load(self, fname):
        self.set_file(fname)
        fabfile = open(self.filepath, "r")
        for line in fabfile.readlines():
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                self.filelines.append(line)
            else:
                kv = line.split("=", 1)
                key = kv[0].strip()
                if len(kv) > 1:
                    value = kv[1].strip()
                else:
                    value = ""
                self.options[key] = value
                self.filelines.append(key)

        fabfile.close()

    def save(self, fpath=None, mask=None):
        """
        Save .fab file. If mask specified use this mask file
        in preference to one in options (used for quick runs)
        """
        if fpath is None:
            fpath = self.filepath
        if not fpath:
            raise RuntimeError("No file specified for save")

        fab = open(fpath, "w")
        self.dump(fab, mask=mask)
        fab.close()

    def dump(self, stream, mask=None):
        """
        Dump to an output stream

        :param stream: Output stream
        :param mask: If specified, replace the mask option with this value. Used for temporary quick runs
        """
        for line in self.filelines:
            if len(line) == 0 or line[0] == "#":
                stream.write(line)
            else:
                if self.options[line] == "":
                    stream.write("%s" % line)
                elif line == "mask" and mask is not None:
                    stream.write("mask=%s" % mask)
                else:
                    stream.write("%s=%s" % (line, self.options[line]))
            stream.write("\n")

    def _init_default(self):
        self.set_file("newfile.fab")

        self.options = {"method": "vb", "noise": "white",
                        "model": "poly", "degree": "2",
                        "output": "out", "overwrite": "",
                        "save-model-fit": ""}
        self.filelines = ["# Generated by fabber_gui.py", "",
                          "output", "overwrite", "save-model-fit", ""
                                                                   "method", "noise", "model", "degree"]

class FabberRun:
    """
    Base class for a completed Fabber run, either from the executable or from the library
    """

    def __init__(self, data, log):
        self.data = data
        self.log = log
        self.timestamp, self.timestamp_str = self.get_log_timestamp(self.log)

    def get_log_timestamp(self, log):
        prefixes = ["start time:", "fabberrundata::start time:"]
        timestamp_str = ""
        for line in log.splitlines():
            l = line.strip()
            for prefix in prefixes:
                if l.lower().startswith(prefix):
                    timestamp_str = l[len(prefix):].strip()
                    try:
                        timestamp = time.strptime(timestamp_str)
                        return timestamp, timestamp_str
                    except:
                        warnings.warn("Failed to parse timestamp: '%s'" % timestamp_str)
        if log != "":
            warnings.warn("Could not find timestamp in log")
        return datetime.datetime.now(), timestamp_str

class Fabber:
    """
    Interface to Fabber in library mode using simplified C-API
    """
    def __init__(self, lib=None, model_libs=[], rundata=None, auto_load_models=False):
        self.ex, def_lib, models = find_fabber()
            
        if lib is not None:
            self.lib = lib
        elif rundata is not None and "fabber_lib" in rundata:
            self.lib = rundata["fabber_lib"]
        else:
            self.lib = def_lib

        if self.lib is None or not os.path.isfile(self.lib):
            raise FabberException("Invalid core library - file not found: %s" % self.lib)

        self.model_libs = set(model_libs)
        if rundata is not None and "loadmodels" in rundata:
            self.model_libs.add(rundata["loadmodels"])
        for lib in self.model_libs:
            if not os.path.isfile(lib):
                raise FabberException("Invalid models library - file not found: %s" % lib)
 
        if auto_load_models:
            for model in models:
                self.model_libs.add(model)
           
        self.errbuf = create_string_buffer(255)
        self.outbuf = create_string_buffer(1000000)
        self.progress_cb_type = CFUNCTYPE(None, c_int, c_int)
        self._init_clib()

    def get_methods(self):
        """ Get known inference methods"""
        self._trycall(self.clib.fabber_get_methods, self.handle, len(self.outbuf), self.outbuf, self.errbuf)
        return self.outbuf.value.splitlines()

    def get_models(self):
        """ Get known models"""
        self._trycall(self.clib.fabber_get_models, self.handle, len(self.outbuf), self.outbuf, self.errbuf)
        return self.outbuf.value.splitlines()

    def get_options(self, method=None, model=None):
        """
        Get known Fabber options

        :param method: If specified, return options for this method
        :param model: If specified, return options for this model

        Only one of method and model should be specified. If neither are specified, generic
        Fabber options are returned.

        :return: Tuple of options, description. Options is a list of options, each in the form of a dictionary.
        Description is a simple text description of the method or model
        """
        if method:
            key = "method"
            value = method
        elif model:
            key = "model"
            value = model
        else:
            key = None
            value = None
        self._trycall(self.clib.fabber_get_options, self.handle, key, value, len(self.outbuf), self.outbuf, self.errbuf)
        opt_keys = ["name", "description", "type", "optional", "default"]
        opts = []
        lines = self.outbuf.value.split("\n")
        for opt in lines[1:]:
            if len(opt) > 0:
                opt = dict(zip(opt_keys, opt.split("\t")))
                opt["optional"] = opt["optional"] == "1"
                opts.append(opt)
        return opts, lines[0]

    def get_model_params(self, rundata):
        """ Get the model parameters, given the specified options"""
        self._init_clib()
        for key, value in rundata.items():
            self._trycall(self.clib.fabber_set_opt, self.handle, str(key), str(value), self.errbuf)

        self._trycall(self.clib.fabber_get_model_params, self.handle, len(self.outbuf), self.outbuf, self.errbuf)

        return self.outbuf.value.splitlines()

    def get_model_outputs(self, rundata=None):
        """ Get additional model timeseries outputs, given the specified options"""
        if rundata is not None:
            self._init_clib()
            for key, value in rundata.items():
                self._trycall(self.clib.fabber_set_opt, self.handle, str(key), str(value), self.errbuf)

        self._trycall(self.clib.fabber_get_model_outputs, self.handle, len(self.outbuf), self.outbuf, self.errbuf)
        return self.outbuf.value.splitlines()

    def model_evaluate(self, rundata, params, nt, indata=None):
        """ """
        self._init_clib()
        for key, value in rundata.items():
            self._trycall(self.clib.fabber_set_opt, self.handle, str(key), str(value), self.errbuf)

        # Get model parameter names
        self._trycall(self.clib.fabber_get_model_params, self.handle, len(self.outbuf), self.outbuf, self.errbuf)
        model_params = self.outbuf.value.splitlines()
        if len(params) != len(model_params):
            raise FabberException("Incorrect number of parameters specified: expected %i (%s)" % (len(model_params), ",".join(model_params)))
        plist = []
        for p in model_params:
            if p not in params:
                raise FabberException("Model parameter %s not specified" % p)
            else:
                plist.append(params[p])
        ret = np.zeros([nt,], dtype=np.float32)
        if indata is None: indata = np.zeros([nt,], dtype=np.float32)
        self._trycall(self.clib.fabber_model_evaluate, self.handle, len(plist), np.array(plist, dtype=np.float32), nt, indata, ret, self.errbuf)

        return ret

    def run(self, rundata, progress_cb=None):
        """
        Run fabber on the provided rundata options

        :param rundata: FabberRunData instance
        :param progress_cb: Callable which will be called periodically during processing
        :return: On success, a FabberRun instance
        """
        mask = None
        data = {}
        for key, value in rundata.items():
            try:
                f = nib.load(value)
                d = f.get_data()
                if key == "mask":
                    mask = d
                else:
                    data[value] = d
            except:
                if key == "data":
                    raise
                # Otherwise ignore, most options will not be data files
                pass

        return self.run_with_data(rundata, data, mask, progress_cb)

    def run_with_data(self, rundata, data, mask=None, progress_cb=None):
        """
        Run fabber

        :param data: Dictionary of data: string key, Numpy array value
        :param mask: Mask as Numpy array, or None if no mask
        :param progress_cb: Callable which will be called periodically during processing
        :return: On success, a FabberRun instance
        """
        if not data.has_key("data"):
            raise Exception("Main voxel data not provided")
        s = data["data"].shape
        nv = s[0] * s[1] * s[2]

        if mask is None: mask = np.ones(nv)
        # Make suitable for passing to int* c function
        mask = np.ascontiguousarray(mask.flatten(order='F'), dtype=np.int32)

        self._init_clib()
        for key, value in rundata.items():
            self._trycall(self.clib.fabber_set_opt, self.handle, str(key), str(value), self.errbuf)
        self._trycall(self.clib.fabber_get_model_params, self.handle, len(self.outbuf), self.outbuf, self.errbuf)
        params = self.outbuf.value.splitlines()

        output_items = []
        if "save-mean" in rundata:
            output_items += ["mean_" + p for p in params]
        if "save-std" in rundata:
            output_items += ["std_" + p for p in params]
        if "save-zstat" in rundata:
            output_items += ["zstat_" + p for p in params]
        if "save-noise-mean" in rundata:
            output_items.append("noise_means")
        if "save-noise-std" in rundata:
            output_items.append("noise_stdevs")
        if "save-free-energy" in rundata:
            output_items.append("freeEnergy")
        if "save-model-fit" in rundata:
            output_items.append("modelfit")
        if "save-residuals" in rundata:
            output_items.append("residuals")
        if "save-mvn" in rundata:
            output_items.append("finalMVN")
        if "save-model-extras" in rundata:
            output_items += self.get_model_outputs()

        retdata, log = {}, ""
        self._trycall(self.clib.fabber_set_extent, self.handle, s[0], s[1], s[2], mask, self.errbuf)
        for key, item in data.items():
            if len(item.shape) == 3:
                size = 1
            else:
                size = item.shape[3]
            item = np.ascontiguousarray(item.flatten(order='F'), dtype=np.float32)
            self._trycall(self.clib.fabber_set_data, self.handle, key, size, item, self.errbuf)

        progress_cb_func = self.progress_cb_type(0)
        if progress_cb is not None:
            progress_cb_func = self.progress_cb_type(progress_cb)

        self._trycall(self.clib.fabber_dorun, self.handle, len(self.outbuf), self.outbuf, self.errbuf, progress_cb_func)
        log = self.outbuf.value
        for key in output_items:
            size = self._trycall(self.clib.fabber_get_data_size, self.handle, key, self.errbuf)

            arr = np.ascontiguousarray(np.empty(nv * size, dtype=np.float32))
            self._trycall(self.clib.fabber_get_data, self.handle, key, arr, self.errbuf)
            if size > 1:
                arr = arr.reshape([s[0], s[1], s[2], size], order='F')
            else:
                arr = arr.reshape([s[0], s[1], s[2]], order='F')
            retdata[key] = arr

        return FabberRun(retdata, log)

    def __del__(self):
        self._destroy_handle()

    def _destroy_handle(self):
        if hasattr(self, "handle"):
            handle = getattr(self, "handle")
            if handle is not None:
                self.clib.fabber_destroy(handle)
                self.handle = None

    def _init_clib(self):
        """
        This is required because currently there is no CAPI function to clear the rundata
        of options
        """
        try:
            self._destroy_handle()
            self.clib = CDLL(str(self.lib))

            # Signatures of the C functions
            c_int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
            c_float_arr = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

            self.clib.fabber_new.argtypes = [c_char_p]
            self.clib.fabber_new.restype = c_void_p
            self.clib.fabber_load_models.argtypes = [c_void_p, c_char_p, c_char_p]
            self.clib.fabber_set_extent.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_int_arr, c_char_p]
            self.clib.fabber_set_opt.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
            self.clib.fabber_set_data.argtypes = [c_void_p, c_char_p, c_uint, c_float_arr, c_char_p]
            self.clib.fabber_get_data_size.argtypes = [c_void_p, c_char_p, c_char_p]
            self.clib.fabber_get_data.argtypes = [c_void_p, c_char_p, c_float_arr, c_char_p]
            self.clib.fabber_dorun.argtypes = [c_void_p, c_uint, c_char_p, c_char_p, self.progress_cb_type]
            self.clib.fabber_destroy.argtypes = [c_void_p]

            self.clib.fabber_get_options.argtypes = [c_void_p, c_char_p, c_char_p, c_uint, c_char_p, c_char_p]
            self.clib.fabber_get_models.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            self.clib.fabber_get_methods.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]

            self.clib.fabber_get_model_params.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            self.clib.fabber_get_model_outputs.argtypes = [c_void_p, c_uint, c_char_p, c_char_p]
            self.clib.fabber_model_evaluate.argtypes = [c_void_p, c_uint, c_float_arr, c_uint, c_float_arr, c_float_arr, c_char_p]
        except Exception, e:
            raise FabberException("Error initializing Fabber library: %s" % str(e))

        self.handle = self.clib.fabber_new(self.errbuf)
        if self.handle is None:
            raise FabberException("Error creating fabber context (%s)" % self.errbuf.value)

        for lib in self.model_libs:
            self._trycall(self.clib.fabber_load_models, self.handle, lib, self.errbuf)

    def _trycall(self, call, *args):
        ret = call(*args)
        if ret < 0:
            raise FabberException(self.errbuf.value, ret, self.outbuf.value)
        else:
            return ret

class FabberLib(Fabber):
    """ For compatibility only"""
    pass
