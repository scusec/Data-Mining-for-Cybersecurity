import re
import numpy as np
from collections import Counter
from get_sinks import get_sinks_from_content, get_sinks_from_file



# 特征抽取helper
def get_index_list(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(int(index))
        index = s.find(sub,index + 1)
    if len(index_list) > 0:
        return index_list
    else:
        return -1


def get_freq_dict(sink):
    freq_dict = dict(Counter(sink))
    return freq_dict


# ************************* 特征抽取 *************************

def dollar_index(sink):
    sink = str(sink).lower()
    return sink.find('$')


def length_of_sink(sink):
    return len(str(sink).lower())


def count_angle(sink):
    return str(sink).lower().count('<')



def count_brace(sink):
    return str(sink).lower().count('{')


def count_double_quotes(sink):
    return str(sink).lower().count('\"')


def count_single_quotes(sink):
    return str(sink).lower().count('\'')


def count_slash(sink):
    return str(sink).lower().count('\/')
    
    
def count_anti_slash(sink):
    return str(sink).lower().count('\\')


def count_parenthesis(sink):
    return str(sink).lower().count('(')


def count_colon(sink):
    return str(sink).lower().count(':')


def dollar_inside_angle(sink):
    left_paren_pos = get_index_list('<', sink)
    if left_paren_pos == -1:
        return -1
    right_paren_pos = get_index_list('>', sink)
    if right_paren_pos == -1:
        return -1
    dollar_pos = get_index_list('$', sink)
    if dollar_pos == -1:
        return -1
    for left, right in zip(left_paren_pos, right_paren_pos):
#         print(left)
#         print(right)
        for dollar in dollar_pos:
#             print(dollar)
            if left < dollar and dollar < right:
                return 1
    return -1
    
    
def dollar_inside_paren(sink):
    left_paren_pos = get_index_list('(', sink)
    if left_paren_pos == -1:
        return -1
    right_paren_pos = get_index_list(')', sink)
    if right_paren_pos == -1:
        return -1
    dollar_pos = get_index_list('$', sink)
    if dollar_pos == -1:
        return -1
    for left, right in zip(left_paren_pos, right_paren_pos):
#         print(left)
#         print(right)
        for dollar in dollar_pos:
#             print(dollar)
            if left < dollar and dollar < right:
                return 1
    return -1


def dollar_inside_brace(sink):
    left_brace_pos = get_index_list('{', sink)
    if left_brace_pos == -1:
        return -1
    right_brace_pos = get_index_list('}', sink)
    if right_brace_pos == -1:
        return -1
    dollar_pos = get_index_list('$', sink)
    if dollar_pos == -1:
        return -1
    for left, right in zip(left_brace_pos, right_brace_pos):
#         print(left)
#         print(right)
        for dollar in dollar_pos:
#             print(dollar)
            if left < dollar and dollar < right:
                return 1
    return -1


def dollar_behind_colon(sink):
    colon_pos = get_index_list(':', sink)
    if colon_pos == -1:
        return -1
    dollar_pos = get_index_list('$', sink)
    if dollar_pos == -1:
        return -1
    for colo in colon_pos:
        for dollar in dollar_pos:
            if colo < dollar:
                return 1
    return -1

feature_func_list = [ dollar_index, count_angle, count_brace,
                      count_double_quotes, count_single_quotes, count_slash,
                      count_anti_slash, count_parenthesis, dollar_inside_angle,
                      dollar_inside_brace, dollar_inside_paren, count_colon, 
                      dollar_behind_colon
                    ]


def sub_letters(sink):
    from re import sub
    sink = sink.lower()
    return sub(pattern=r'\w*', repl='', string=sink) 


def get_sink_features(sink):
    try:
        sink = str(sink)
    except Exception:
        print('Input parameter cannot be explained as a sink string!')
    else:
        feature_list = []
        sink = (sub_letters(sink))  # Get rid of all letters in the sink, focus on the symbols
        for func in feature_func_list:
            feature_list.append(func(sink))
        if len(feature_list) != 0:
            return feature_list
        else:
            return []


