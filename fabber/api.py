"""
PYFAB API
=========

PYFAB can interface with Fabber in two ways:

 - Via a wrapper of the command line tools (generally one per model library, e.g. ``fabber_asl``,
   ``fabber_cest``, etc)
 - By accessing the C API directly using the core shared library ``libfabbercore_shared.so``
   (in conjunction with model libraries such as ``libfabber_models_asl.so``).

PYFAB can use either approach. The shared library is more elegant and probably a little faster
for very short runs, however the shared library is not available in all installations - in
particular it is not currently included in FSL.

The FabberApi interface described below is implemented for each option. So to work with the command line
wrapper you would use, for example::

    >>> from fabber import FabberCl
    >>> fab = FabberCl()
    >>> fab.get_models(model_group="asl")
    [u'asl_2comp', u'asl_multiphase', u'aslrest', u'buxton', u'linear', u'poly', u'quasar', u'satrecov', u'turboquasar']

Almost identically, to do the same thing using the shared library API::

    >>> from fabber import FabberShlib
    >>> fab = FabberShlib()
    >>> fab.get_models(model_group="asl")
    [u'asl_2comp', u'asl_multiphase', u'aslrest', u'buxton', u'linear', u'poly', u'quasar', u'satrecov', u'turboquasar']

For convenience the generic class Fabber is defined to be FabberShlib where a shared core library 
exists, and FabberCl otherwise. So if you don't want to have to worry about which to use, just do::

    >>> from fabber import Fabber
    >>> fab = Fabber()
    >>> fab.get_models(model_group="asl")
    [u'asl_2comp', u'asl_multiphase', u'aslrest', u'buxton', u'linear', u'poly', u'quasar', u'satrecov', u'turboquasar']
"""

import os
import sys
import warnings
import datetime
import time
import glob
import re
import collections
import tempfile

import numpy as np
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
    def _progress(done, total):
        complete = 100*done/total
        log.write("\b\b\b\b%3i%%" % complete)
        log.flush()
    return _progress

def _find_file(current_value, search_dir, search_for):
    if current_value is not None:
        return current_value
    else:
        newfpath = os.path.join(search_dir, search_for)
        if os.path.isfile(newfpath):
            return newfpath
        else:
            return current_value

def find_fabber(*extra_search_dirs, **kwargs):
    """
    Find the Fabber executable, core library and model libraries, or return None if not found

    :param search_dirs: Extra search directories to use to look for Fabber libraries and executables. By default will
                        use the environment variables ``FABBERDIR``, ``FSLDEVDIR`` and ``FSLDIR`` in that order.

    :return: A tuple of: core library, core executable, dictionary of model group libraries, 
             dictionary of model group executables
    """
    debug = kwargs.get("debug", False)
    exe, lib, model_libs, model_exes = None, None, {}, {}
    search_dirs = list(extra_search_dirs)
    for envdir in ("FABBERDIR", "FSLDEVDIR", "FSLDIR"):
        if envdir in os.environ:
            search_dirs.append(os.environ[envdir])

    if debug:
        sys.stderr.write("Search dirs: %s\n" % search_dirs)

    for search_dir in search_dirs:
        exe = _find_file(exe, search_dir, _BIN_FORMAT % "fabber")
        lib = _find_file(lib, search_dir, _LIB_FORMAT % "fabbercore_shared")
        lib_regex = re.compile(r'.*fabber_models_(.+)\..*')
        for model_lib in glob.glob(os.path.join(search_dir, _LIB_FORMAT % "fabber_models_*")):
            group_name = lib_regex.match(model_lib).group(1).lower()
            model_libs[group_name] = model_libs.get(group_name, model_lib)
        exe_regex = re.compile(r'.*fabber_(.+)\.?.*')
        for model_exe in glob.glob(os.path.join(search_dir, _BIN_FORMAT % "fabber_*")):
            group_name = exe_regex.match(model_exe).group(1).lower()
            if debug:
                sys.stderr.write("EXE: %s\n" % group_name)
            if group_name != "var":
                # Old executable which messes things up sometimes!
                model_exes[group_name] = model_exes.get(group_name, model_exe)
                if debug:
                    sys.stderr.write("fname: %s\n" % model_exes[group_name])

    return lib, exe, model_libs, model_exes

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
    Dump options to an output stream

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

    :ivar errcode: Error code from Fabber
    :ivar log: Log output if available

    The error string (if available) is passed to the RuntimeError
    constructor.
    """
    def __init__(self, msg, errcode=None, log=None):
        self.errcode = errcode
        self.log = log
        self.msg = msg
        if errcode is not None:
            RuntimeError.__init__(self, "%i: %s" % (errcode, msg))
        else:
            RuntimeError.__init__(self, msg)

    def __str__(self):
        if self.errcode is not None:
            return "FabberException: %i: %s" % (self.errcode, self.msg)
        else:
            return "FabberException: %s" % self.msg

class FabberRun(object):
    """
    The result of a Fabber run

    :ivar data: A mapping from output name (e.g. ``mean_ftiss`` to data object)
    :ivar log: Log output from fabber as a string
    :ivar timestamp: Timestamp of the run as a Python datetime object
    :ivar timestamp_str: String version of the timestamp
    """
    def __init__(self, data, log):
        self.data = data
        self.log = log
        self.timestamp, self.timestamp_str = self._get_log_timestamp(self.log)
        self.ref_nii = None

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

        if ref_nii is None:
            ref_nii = self.ref_nii

        if ref_nii is not None:
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

class FabberApi(object):
    """
    Abstract interface to Fabber, which can be implemented either using the shared library or 
    command line wrapper
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

    def __init__(self, core_lib=None, model_libs=None, core_exe=None, model_exes=None, **kwargs):
        self._debug = kwargs.get("debug", False)
        self.core_lib, self.core_exe, self.model_libs, self.model_exes = find_fabber()

        if core_lib:
            self.core_lib = core_lib

        if core_exe:
            self.core_exe = core_exe

        if model_libs is not None:
            self.model_libs = dict(model_libs)

        for lib in self.model_libs.values():
            if not os.path.isfile(lib):
                raise FabberException("Invalid models library - file not found: %s" % lib)

        if model_exes is not None:
            self.model_exes = dict(model_exes)

        for exe in self.model_exes.values():
            if not os.path.isfile(exe):
                raise FabberException("Invalid models executable - file not found: %s" % exe)

    def get_model_groups(self):
        """
        Get known model groups

        :return: Sequence of model group names
        """
        return list(self.model_libs.keys()) + list(self.model_exes.keys())

    def get_methods(self):
        """
        Get known inference methods

        :return: Sequence of known inference method names
        """
        pass

    def get_models(self, model_group=None):
        """
        Get known models

        :param model_group: If specified, return only models in this group
        :return: Sequence of known model names
        """
        pass

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
        pass

    def get_model_params(self, options):
        """
        Get model parameters

        :param options: Options dictionary
        :return: Sequence of model parameter names
        """
        pass

    def get_model_param_descs(self, options):
        """
        Get model parameters and descriptions.

        This is an extension to ``get_model_params``
        which returns not only the parameter names but their descriptions and units as
        well (if the model provides them - many do not!)

        :param options: Options dictionary
        :return: Sequence of tuples each containing parameter name, description and units.
                 If either of description or units were not provided, empty strings are
                 returned in their place.
        """
        pass

    def get_model_outputs(self, options):
        """
        Get additional model timeseries outputs

        :param options: Fabber options
        :return: Sequence of names of additional model timeseries outputs
        """
        pass

    def model_evaluate(self, options, param_values, nvols, indata=None, output_name=""):
        """
        Evaluate the model with a specified set of parameters

        :param options: Fabber options as key/value dictionary
        :param param_values: Parameter values as a dictionary of param name : param value
        :param nvols: Length of output array - equivalent to number of volumes in input data set
        """
        pass

    def run(self, options, progress_cb=None):
        """
        Run fabber

        :param options: Fabber options as key/value dictionary. Data may be passed as Numpy arrays, Nifti
                        images or strings (which are interpreted as filenames)
        :param progress_cb: Callable which will be called periodically during processing. It will be passed
                            two numeric parameters, the first a measure of work done and the second a measure
                            of total work. These should not be interpreted any further! They may be counts
                            of voxels, iterations or any other arbitrary units.

        :return: On success, a FabberRun instance
        """
        pass

    def is_data_option(self, key, options):
        """
        :param key: Option name
        :param options: Known model options as key/value dict

        :return: True if ``key`` is a voxel data option
        """
        if key in ("data", "mask", "suppdata", "continue-from-mvn"):
            return True
        elif key.startswith("PSP_byname") and key.endswith("_image"):
            return True
        else:
            return key in [option["name"] for option in options
                           if option["type"] in (self.IMAGE, self.TIMESERIES, self.MVN)]

    def is_matrix_option(self, key, options):
        """
        :param key: Option name
        :param option: Known model options as key/value dict

        :return True if ``key`` is a matrix option
        """
        return key in [option["name"] for option in options if option["type"] == self.MATRIX]

    def is_sequence_option(self, key, options):
        """
        :param key: Option name
        :param option: Known model options as key/value dict

        :return True if ``key`` matches a sequence option. This is the case if a known model
                option contains ``<n>`` and, when this is removed, the remainder matches the
                supplied key. For example if the model defines an option ``pld<n>`` and the user
                supplies an option with the key ``pld`` then (if the value is a sequence) this 
                can be expanded into --pld1=<val1> --pld2=<val2> etc. 
        """
        return key in [option["name"].replace("<n>", "") for option in options if "<n>" in option["name"]]

    def _write_temp_matrix(self, matrix, tempdir=None):
        """
        Write a Numpy array to a temporary file as an ASCII matrix

        :param matrix: 2D Numpy array
        :param tempdir: Optional temporary directory

        :return: Name of temporary file
        """
        with tempfile.NamedTemporaryFile(prefix="fab", delete=False, dir=tempdir, mode="wt") as tempf:
            for row in matrix:
                if isinstance(row, collections.Sequence):
                    tempf.write(" ".join(["%.6g" % val for val in row]) + "\n")
                else:
                    tempf.write("%f\n" % row)
            return tempf.name

    def _write_temp_nifti(self, nii, tempdir=None):
        """
        Write a Nifti data set to a temporary file

        :param nii: ``nibabel.Nifti1Image``
        :param tempdir: Optional temporary directory

        :return: Name of temporary file
        """
        with tempfile.NamedTemporaryFile(prefix="fab", delete=False, dir=tempdir, mode="wt") as tempf:
            name = tempf.name
            tempf.close()
            nii.to_filename(name)
            return name

    def _normalize_options(self, options):
        """
        Prepare option keys/values for passing to fabber:

         - Options with value of None are ignored

         - Key separators can be specified as underscores or hyphens as hyphens are not allowed in Python
           keywords. They are always passed as hyphens except for the anomolous PSP_byname options

         - Fabber interprets boolean values as 'option given=True, not given=False'. For options with the
           value True, the actual option value passed must be blank

         - If option value is a list or tuple (ONLY these types) it will automatically expand out into
           Fabber-style list options, e.g. ti=[0.5, 1.0, 1.5] will generate ti1=0.5, ti2=1.0, ti3=1.5
        """
        options_normalized = {}
        for key, value in options.items():
            if not key.startswith("PSP_"):
                key = key.replace("_", "-")

            if value is None:
                key, value = None, None
            elif isinstance(value, bool):
                if value:
                    value = ""
                else:
                    key, value = None, None

            if key is not None:
                options_normalized[key] = value
        return options_normalized

    def _parse_params(self, lines):
        """
        Parse the model parameter output
        """
        params = []
        for line in lines:
            if line.strip():
                # Format of --listparams is <name> [<desc> [(units: <units>)]]
                parts = line.split(" ", 1)
                param = [parts[0], "", ""]
                if len(parts) > 1:
                    units_match = re.search(r"(.+?)(?:\s\(units: (.*)\))?$", parts[1])
                    if units_match:
                        param[1] = units_match.group(1)
                        param[2] = units_match.group(2)
                    else:
                        param[1] = parts[1]
                params.append(param)
        # FIXME temp to avoid incompatibility when units are not expected
        return [params[0] for param in params]
