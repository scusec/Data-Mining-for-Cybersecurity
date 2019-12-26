import re
import numpy as np

# ************************* 接口 *************************
def get_sinks_from_file(file_path):
    code_content = ''
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        code_content = f.read()
    if len(code_content) == 0:
        print('Read code file error')
        return []
    else:
        sink_list = get_sinks_from_content(code_content)
        return sink_list


def get_sinks_from_content(code_content):
    '''
    Extract sinks from code contents
    returns a list of sinks
    '''
    ptrn = r'echo\s(.+)\s*;'
    sink_list = re.findall(pattern=ptrn, string=code_content)
    if sink_list == []:
        print('No sink in this file.')
    if type(sink_list) == 'str':
        sink_list = [sink_list]
    return sink_list


