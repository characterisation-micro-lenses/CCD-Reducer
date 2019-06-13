import sys

class Error(Exception):
    """Generates Error Classes"""
    def __init__(self, message=None):
        if message is None:
            super(Error).__init__()
        else:
            super(Error, self).__init__(message)

        tb = sys.exc_info()[2]
        self.with_traceback(tb)
