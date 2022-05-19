import scipy
import sys
import os
import sys
import platform
import pandas
import pytz
cur_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(cur_dir)
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))
if platform.system() == 'Linux':
	sys.path.append('/home/users/ntu/zhihaoqu/.conda/envs/mypyenv')
from CrossPlatformDev import my_print, join_str
f = open('print_log.txt', 'a')
if platform.system() == 'Windows':
	env = os.environ['USERPROFILE']
elif platform.system() == 'Linux':
	env = os.environ['HOME']
print(env, file=f)
print('pytz vers:'+pytz.__version__, file=f)
print('pytz file: '+pytz.__file__, file=f)
print('scipy vers: '+scipy.__version__, file=f)
print('scipy file: '+scipy.__file__, file=f)
print('python vers: '+sys.version, file=f) 
print('pandas vers: '+pandas.__version__, file=f)
print('pandas file: '+pandas.__file__, file=f)
print('importing join_str, test: '+ join_str('a', 'b'), file=f)
f.close()