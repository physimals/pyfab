"""
Python API for the FSL Fabber tool
"""

from .api import FabberException, FabberRun, percent_progress, find_fabber
from .api_shlib import FabberShlib
from .api_cl import FabberCl
from .model_test import self_test, generate_test_data

def Fabber(*search_dirs):
    """
    Get an API object for Fabber. Uses the shared lib API if available
    otherwise falls back to the command line

    :param extra_search_dirs: Extra search directories to use to look for Fabber libraries and executables
    """
    corelib, coreexe, libs, exes = find_fabber(*search_dirs)
    if corelib:
        return FabberShlib(core_lib=corelib, model_libs=libs)
    else:
        return FabberCl(core_exe=coreexe, model_exes=exes)
        
__all__ = ["Fabber", "FabberException", "FabberRun", "self_test", "generate_test_data", "percent_progress"]
