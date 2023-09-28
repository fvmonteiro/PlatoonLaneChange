# Solution from:
# https://stackoverflow.com/questions/47381835/scipy-minimize-get-cost-function-vs-iteration
# Attempt to get some sort of cost vs iteration information.
# After tests, it seems like the chosen optimization method (SLSQP) does not
# display any meaningful intermediary outputs anyway.

import os
import sys
import tempfile
from typing import Union


class forcefully_redirect_stdout(object):
    """ Forces stdout to be redirected, for both python code and C/C++/Fortran
        or other linked libraries.  Useful for scraping values from e.g. the
        disp option for scipy.optimize.minimize.
    """

    def __init__(self, to: Union[str, None] = None):
        """ Creates a new forcefully_redirect_stdout context manager.

        :param to: what to redirect to.  If type(to) is None, internally uses
         a tempfile.SpooledTemporaryFile and returns a UTF-8
         string containing the captured output.  If type(to) is str, opens a
         file at that path and pipes output into it, erasing prior contents.

        :returns: `str` if type(to) is None, else returns `None`.

        """

        # initialize where we will redirect to and a file descriptor for python
        # stdout -- sys.stdout is used by python, while os.fd(1) is used by
        # C/C++/Fortran/etc
        self.to = to
        self.fd = sys.stdout.fileno()
        if self.to is None:
            self.to = tempfile.SpooledTemporaryFile(mode='w+b')
        else:
            self.to = open(to, 'w+b')

        self.old_stdout = os.fdopen(os.dup(self.fd), 'w')
        self.captured = ''

    def __enter__(self):
        self._redirect_stdout(to=self.to)
        return self

    def __exit__(self, *args):
        self._redirect_stdout(to=self.old_stdout)
        self.to.seek(0)
        self.captured = self.to.read().decode('utf-8')
        self.to.close()

    def _redirect_stdout(self, to):
        sys.stdout.close()  # implicit flush()
        os.dup2(to.fileno(), self.fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(self.fd, 'w')  # Python writes to fd

# Usage example:
# import re
# from scipy.optimize import minimize
#
# def foo(x):
#     return 1/(x+0.001)**2 + x
#
# with forcefully_redirect_stdout() as txt:
#     result = minimize(foo, [100], method='L-BFGS-B', options={'disp': True})
#
# print('this appears before `disp` output')
# print('here''s the output from disp:')
# print(txt.captured)
# lines_with_cost_function_values = \
#     re.findall(r'At iterate\s*\d\s*f=\s*-*?\d*.\d*D[+-]\d*', txt.captured)
#
# fortran_values = [s.split()[-1] for s in lines_with_cost_function_values]
# # fortran uses "D" to denote double and "raw" exp notation,
# # fortran value 3.0000000D+02 is equivalent to
# # python value  3.0000000E+02 with double precision
# python_vals = [float(s.replace('D', 'E')) for s in fortran_values]
# print(python_vals)
