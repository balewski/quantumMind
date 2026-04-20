import os
import json
from pprint import pprint


#...!...!....................
def env_2_dict(name, verb=1):
    tmpJ=os.environ[name]
    tmpD=json.loads(tmpJ)
    if verb>0:
        print('\n----- %s dump:'%name)
        pprint(tmpD)
    return tmpD



#...!...!....................
def dump_file(name):
    print('\nDump '+name)
    # Open the file for reading
    with open(name, "r") as file:
        # Read and print the entire file content
        file_content = file.read()
        print(file_content)


#...!...!....................
def util_func1():
    print('I AM util_func1')

