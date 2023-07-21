from sys import platform


# tcp setup
ADDR = '192.168.0.84'
RECV_PORT = 9105
SEND_PORT = 9106

# logging setup
DEPLOY_OUTPUT_DIR = '/home/user/.deployment/TimeSeriesClassifier/output'
if platform in ['linux', 'linux2']:
    OUTPUT_DIR = '/home/user/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output'

elif platform == 'darwin':
    OUTPUT_DIR = '/Users/jeonghyeonho/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output'

elif platform == 'win32':
    raise NotImplementedError
