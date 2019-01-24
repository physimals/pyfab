"""
PYFAB API using Fabber command line tool(s)
===========================================

This API uses Fabber command line tools, run within temporary
directories, to implement the required functions. For short
runs this is likely to be slower than the shared-library
based API however it can be used when the shared library is not 
available or there are binary compatibility problems.

Note that Fabber command line tools typically come as one
per model library, i.e. ``fabber_asl`` which contains a number
of ASL-MRI related models, ``fabber_cest`` which contains the
CEST-MRI model, etc. The base ``fabber`` executable is not 
terribly useful in itself - it only includes generic linear
and polynomial models.
"""
from __future__ import unicode_literals, print_function

import collections
import tempfile
import subprocess
import shutil
import re
import os
import glob
import warnings

import six
import numpy as np
import nibabel as nib

from .api import FabberApi, FabberException, FabberRun

def _progress_stdout_handler(progress_cb):
    """
    :return: stdout handler which looks for percentage done reports
             and makes an appropriate call to the progress callback.

    The handler uses a regex to look for a 'percentage' 
    output and calls the progress handler with the number
    found and a 'total' of 100.
    """
    percent_re = re.compile(r"^(\d+)%?$")
    def _handler(line):
        match = percent_re.match(line)
        if match:
            progress_cb(int(match.group(1)), 100)
    return _handler

class FabberClException(FabberException):
    """
    Exception originating from the command line

    We try to read the logfile and also attempt to 
    determine the message from the stdout
    """
    def __init__(self, stdout, returncode, outdir):
        """
        :param stdout: Standard output/error combined
        :param returncode: Return code from executable
        :param outdir: Output directory (to read logfile if possible)
        """
        grabnext = False
        msg = ""
        for line in stdout.splitlines():
            if grabnext:
                msg = line.strip()
                grabnext = False
            if line.startswith("Exception"):
                grabnext = True
        logfile = os.path.join(outdir, "logfile")
        log = ""
        if os.path.exists(logfile):
            with open(logfile, "r") as logfile: log = logfile.read()
        
        FabberException.__init__(self, msg, returncode, log)

class FabberClRun(FabberRun):
    """
    Run output from the command line API

    Sets the attributes log, timestamp, timestamp_str and data
    """
    
    def __init__(self, outdir):
        """
        :param outdir: Directory containing Fabber output
        """
        with open(os.path.join(outdir, "logfile"), "r") as logfile:
            log = logfile.read()
            
        data = {}
        alphanum = "[a-zA-Z0-9_]"
        regexes = [
            re.compile(r".*[/\\](mean_%s+)\..+" % alphanum),
            re.compile(r".*[/\\](std_%s+)\..+" % alphanum),
            re.compile(r".*[/\\](zstat_%s+)\..+" % alphanum),
            re.compile(r".*[/\\](noise_means)\..+"),
            re.compile(r".*[/\\](noise_stdevs)\..+"),
            re.compile(r".*[/\\](finalMVN)\..+"),
            re.compile(r".*[/\\](freeEnergy)\..+"),
            re.compile(r".*[/\\](modelfit)\..+"),
        ]
        for fname in glob.glob(os.path.join(outdir, "*")):
            for regex in regexes:
                match = regex.match(fname)
                if match: data[match.group(1)] = nib.load(fname).get_data()

        FabberRun.__init__(self, data, log)

class FabberCl(FabberApi):
    """
    Interface to Fabber using command line
    """

    def __init__(self, core_exe=None, model_exes=None):
        FabberApi.__init__(self, core_exe=core_exe, model_exes=model_exes)

        if core_exe is not None and not os.path.isfile(self.core_exe):
            raise FabberException("Invalid core executable - file not found: %s" % self.core_exe)

        self._model_groups = None
        self._models = None

    def get_methods(self):
        stdout = self._call(listmethods=True)
        return stdout.splitlines()

    def get_models(self, model_group=None):
        if self._model_groups is None:
            self._model_groups = {}
            self._models = {}
            for group in self.model_exes:
                stdout = self._call(listmodels=True, model_group=group)
                self._models[group] = stdout.splitlines()
                for model in self._models[group]:
                    self._model_groups[model] = group
        if model_group is not None:
            return self._models[model_group]
        else:
            return list(self._model_groups.keys())
       
    def get_options(self, generic=None, method=None, model=None):
        if generic is None:
            # For backwards compatibility - no params = generic
            generic = not method and not model

        ret, all_lines = [], []
        if method:
            stdout = self._call(help=True, method=method)
            lines = stdout.splitlines()
            ret.append(lines[0])
            all_lines += lines[1:]
        if model:
            stdout = self._call(help=True, model=model)
            lines = stdout.splitlines()
            ret.append(lines[0])
            all_lines += lines[1:]
        if generic:
            stdout = self._call(help=True)
            lines = stdout.splitlines()
            ret.append(lines[0])
            all_lines += lines[1:]
        
        opts = self._parse_options(all_lines)
        ret.insert(0, opts)
        return tuple(ret)

    def get_model_params(self, options):
        stdout = self._call(options, listparams=True, data_options=True)
        return stdout.splitlines()
        
    def get_model_outputs(self, options):
        stdout = self._call(options, listoutputs=True, data_options=True)
        return stdout.splitlines()

    def model_evaluate(self, options, param_values, nvols, indata=None, output_name=""):
        params = self.get_model_params(options)
        plist = []
        for param in params:
            if param not in param_values:
                raise FabberException("Model parameter %s not specified" % param)
            else:
                plist.append(param_values[param])

        if len(param_values) != len(params):
            raise FabberException("Incorrect number of parameters specified: expected %i (%s)" % (len(params), ",".join(params)))

        params_file = self._write_temp_matrix(plist)
        data_file = None
        if indata is not None:
            data_file = self._write_temp_matrix(indata)

        stdout = self._call(options, evaluate=output_name, evaluate_nt=nvols, evaluate_params=params_file, evaluate_data=data_file)
        ret = []
        for line in stdout.splitlines():
            try:
                val = float(line)
                ret.append(val)
            except:
                warnings.warn("Unexpected output: %s" % line)
        return ret

    def run(self, options, progress_cb=None):
        if "data" not in options:
            raise ValueError("Main voxel data not provided")

        if progress_cb is not None:
            stdout_handler = _progress_stdout_handler(progress_cb)
        else:
            stdout_handler = None

        outdir = None
        try:
            outdir = tempfile.mkdtemp("fabberout")
            out_subdir = os.path.join(outdir, "fabout")
            self._call(options, output=out_subdir, stdout_handler=stdout_handler, simple_output=True, data_options=True)
            return FabberClRun(out_subdir)
        finally:
            if outdir is not None:
                shutil.rmtree(outdir)

    def _parse_options(self, lines):
        """
        Parse option specifiers like:
        
            --save-mean [BOOL,NOT REQUIRED,NO DEFAULT]
            Output the parameter means.
        """
        options = []
        current_option = None
        option_regex = re.compile(r'--(.+)\s\[(.+),(.+),(.+)]')
        for line in lines:
            match = option_regex.match(line)
            if match:
                if current_option is not None:
                    current_option["description"] = current_option["description"][:-1]

                current_option = {
                    "name" : match.group(1),
                    "type" : match.group(2),
                    "optional" : match.group(3) == 'NOT REQUIRED',
                    "default" : match.group(4),
                    "description" : ""
                }
                if current_option["default"] == "NO DEFAULT":
                    current_option["default"] = ""
                else:
                    current_option["default"] = current_option["default"].split("=", 1)[1]

                options.append(current_option)
            elif current_option is not None:
                desc = line.strip()
                if desc: current_option["description"] += desc + " "

        return options

    def _call(self, options=None, stdout_handler=None, data_options=False, **kwargs):
        """
        Call the Fabber executable
        """
        indir = None
        try:
            if options is None:
                options = {}
            else:
                options = dict(options)
            options.update(kwargs)

            # Convert options to format expected by Fabber (e.g. _ -> -)
            options = self._normalize_options(options)

            # If required, write data options to temporary files
            if data_options:
                indir, options = self._process_data_options(options)

            # Get the correct executable for the model/model group required
            exe = self._get_exe(options)

            # Convert options to command line arguments
            cl_args = self._get_clargs(options)

            # Run the process and return stdout
            stdout = ""
            p = subprocess.Popen([exe] + cl_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while 1:
                retcode = p.poll() # returns None while subprocess is running
                line = p.stdout.readline().decode('utf-8')
                stdout += line
                if stdout_handler is not None:
                    stdout_handler(line)
                if line == "" and retcode is not None: 
                    break

            #print(stdout)
            if retcode != 0:
                raise FabberClException(stdout, retcode, options.get("output", ""))
            return stdout
        finally:
            if indir is not None:
                shutil.rmtree(indir)

    def _get_exe(self, options):
        """
        Get the right Fabber exe to use
        """
        if "model_group" in options or "model-group" in options:
            group = options.pop("model_group", options.pop("model-group", ""))
            if group not in self.model_exes:
                raise ValueError("Unknown model group: %s" % group)
            exe = self.model_exes[group]
        elif "model" in options:
            self.get_models()
            model = options["model"]
            if model not in self._model_groups:
                raise ValueError("Unknown model: %s" % model)
            exe = self.model_exes[self._model_groups[model]]
        elif self.core_exe is not None:
            exe = self.core_exe
        elif self._model_groups:
            exe = self.model_exes.values()[0]
        else:
            raise ValueError("No Fabber executable found")
        return exe

    def _process_data_options(self, options):
        """
        Identify options which need to be written to temporary files before Fabber
        can use them

        :return: Tuple of temp directory name, new options dict
        """
        indir = tempfile.mkdtemp("fabberin")
        try:
            options = dict(options)
            model_options = self.get_options(model=options.get("model", None), method=options.get("method", None), generic=True)[0]
            for key in list(options.keys()):
                if self.is_data_option(key, model_options):
                    # Allow input data to be given as Numpy array, Nifti image or filename. It
                    # must be passed to Fabber as a file name
                    value = options.pop(key)
                    if value is None:
                        pass
                    elif isinstance(value, six.string_types):
                        options[key] = value
                    elif isinstance(value, nib.Nifti1Image):
                        options[key] = self._write_temp_nifti(value, indir)
                    elif isinstance(value, np.ndarray):
                        nii = nib.Nifti1Image(value, affine=np.identity(4))
                        options[key] = self._write_temp_nifti(nii, indir)
                    else:
                        raise ValueError("Unsupported type for input data: %s = %s" % (key, type(value)))

                elif self.is_matrix_option(key, model_options):
                    # Input matrices can be given as Numpy arrays or sequences but must be
                    # passed to fabber as file names
                    value = options.get(key)
                    if value is None:
                        pass
                    elif isinstance(value, six.string_types):
                        options[key] = value
                    elif isinstance(value, (np.ndarray, collections.Sequence)):
                        options[key] = self._write_temp_matrix(value, indir)
                    else:
                        raise ValueError("Unsupported type for input matrix: %s = %s" % (key, type(value)))
            return indir, options
        except:
            shutil.rmtree(indir)
            raise

    def _get_clargs(self, options):
        """
        Build command line arguments from options
        """
        cl_args = []
        for key, value in options.items():
            if key:
                if value != "":
                    cl_args.append("--%s=%s" % (key, value))
                else:
                    cl_args.append("--%s" % key)
                
        return cl_args
