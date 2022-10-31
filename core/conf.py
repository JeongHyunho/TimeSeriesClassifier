from sys import platform


# tcp setup
ADDR = '172.16.0.12'
PORT = 9107 # 0915

# logging setup
if platform in ['linux', 'linux2']:
    OUTPUT_DIR = '/home/user/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output'
    DEPLOY_OUTPUT_DIR = '/home/user/.deployment/TimeSeriesClassifier/output'

elif platform == 'darwin':
    OUTPUT_DIR = '/Users/jeonghyunho/Dropbox/MATLAB_dropbox/TimeSeriesClassifier/output'
    DEPLOY_OUTPUT_DIR = None

elif platform == 'win32':
    raise NotImplementedError
