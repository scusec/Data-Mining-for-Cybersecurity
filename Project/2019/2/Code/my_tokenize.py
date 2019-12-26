# coding = utf-8
#!/usr/bin/env python

import re
from joblib import load
from extract_sink_features import sink_predict_class
import numpy as np


def my_token_get_all(code_file_path):
    '''
    Input: The absolute path of ONE code file.
    Output: The list of tokens of this code.
    '''
    with open(code_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        token_pattern = r'<*[\!\?\$\w\/\'\" \(\)\.\<\>\\\/\=\[\]\-\,\`]+>*'
        tokens = re.findall(pattern=token_pattern, string=content)
        f.close()
    return tokens


def feature_vector_helper(token_get_all_list):
    '''
    Input: A list of tokens of ONE code file.
    Output: Feature vectors list of ONE code file.
    '''

    GLOBALS = [
        '$GLOBALS', '$_SERVER', '$_REQUEST',
        '$_POST', '$_GET', '$_FILES',
        '$_ENV', '$_COOKIE', '$_SESSION'
    ]
    # GLOBALS - A list of PHP Super Global Variables
    
    STOP_TOKENS = []
    # STOP_TOKEN - A list of ignorable variable

    FEATURE_LIST = []
    # FEATURE_LIST - The list of return's features, initially empty
    
    BUILTIN_SANITIZE_FUNCS = [
        'rawurlencode', 'filter_var', 'mysql_real_escape_string',
        'urlencode', 'htmlentities', 'http_build_query', 'htmlspecialchars',
        'preg_replace', 'addslashes'
    ]
    # BUILTIN_SANITIZE_FUNCS - The list of built-in sanitizing functions

    SINK_TYPE_DICT = {
        0: 'T_SINK_BODY_SECTION',
        1: 'T_SINK_SINGLE_QUOTE_ATTR_VAL',
        2: 'T_SINK_DOUBLE_QUOTE_ATTR_VAL',
        3: 'T_SINK_DOUBLE_QUOTE_CSS_VAL',
        4: 'T_SINK_SINGLE_QUOTE_ATTR_VAL',
        5: 'T_SINK_SINGLE_QUOTE_ATTR_VAL',
        6: 'T_SINK_DOUBLE_QUOTE_CSS_VAL',
        7: 'T_SINK_NO_QUOTE_ATTR_VAL',
        8: 'T_SINK_NO_QUOTE_CSS_VAL',
        9: 'T_SINK_SINGLE_QUOTE_JS_BLOCK',
        10: 'T_SINK_DOUBLE_QUOTE_ATTR_VAL',
        11: 'T_SINK_DOUBLE_QUOTE_EVENT_VAL',
        12: 'T_SINK_SINGLE_QUOTE_CSS_VAL',
        13: 'T_SINK_DOUBLE_QUOTE_JS_BLOCK',
        14: 'T_SINK_HTML_TAG_NAME',
        15: 'T_SINK_ATTR_NAME',
        16: 'T_SINK_SINGLE_QUOTE_JS_BLOCK'        
    }
    # SINK_TYPE_DICT = The dictionary of sink type (int) to sink type (string)

    tag_pattern = r' *<.+> *'
    tag_doc_pattern = r' *< *\! *DOCTYPE *(\w+) *> *'
    tag_start_pattern = r'<(\w+)>'
    # tag_end_pattern1 = r' *</(\w+)> *'
    # tag_end_pattern2 = r' *<(\w+)/> *'
    tag_php_start_pattern  = r' *<\?php *'
    tag_php_end_pattern = r' *\?> *'
    var_assign_pattern = r' *\$ *\w+ *.* *= *\S* *'
    var_source_pattern = r' *\$ *\w+ *.* *= *(\$_[A-Z]+) *'
    var_func_pattern = r' *\$ *\w+ *.* *= *(\S+\(.*\)) *'

    for token in token_get_all_list:

        # stop_token? pass:
        if token in STOP_TOKENS:
            # print('T_STOP, pass.')
            continue
        
        elif re.match(pattern=r'//\w+', string=token) != None:
            # print('Useless comment, pass')
            continue
        # <> tag
        elif re.match(pattern=tag_pattern, string=token) != None:

            # <!DOCTYPE xxxx> document type tag
            if len(re.findall(pattern=tag_doc_pattern, string=token.upper())) != 0:
                # print('Found doc tag')
                finds = re.findall(pattern=tag_doc_pattern, string=token.upper())
                doc_type = str(finds[0])
                tk = 'T_TAG_DOCTYPE_' + doc_type.upper()
                FEATURE_LIST.append(tk)

            # <xxx> start tag
            elif len(re.findall(pattern=tag_start_pattern, string=token)) != 0:
                finds = re.findall(pattern=tag_start_pattern, string=token)
                tag_name = str(finds[0]).upper()
                tk = 'T_TAG_' + tag_name + '_START'
                # tk = 'T_TAG_START'
                if tk == 'T_TAG_H1_START':
                    continue
                FEATURE_LIST.append(tk)

            # </xxx> end tag
            elif '/' in token:
                finds = re.findall(pattern=r'\w+', string=token)
                tag_name = str(finds[0]).upper()
                tk = 'T_TAG_' + tag_name + '_END'
                # tk = 'T_TAG_END'
                FEATURE_LIST.append(tk)

            # # <xxx/> end tag
            # elif len(re.findall(pattern=tag_end_pattern2, string=token)) != 0:
            #     finds = re.findall(pattern=tag_end_pattern2, string=token)
            #     tag_name = str(finds).upper()
            #     tk = 'T_TAG_' + tag_name + '_END'
            #     FEATURE_LIST.append(tk)

            # Unknown tag type
            else:
                continue
                # FEATURE_LIST.append('T_TAG_UNKNOWN')


        # <?php php start tag
        elif re.match(pattern=tag_php_start_pattern, string=token.lower()) != None:
            tk = 'T_TAG_PHP_START'
            # tk = 'T_TAG_START'
            FEATURE_LIST.append(tk)
        elif re.match(pattern=tag_php_end_pattern, string=token.lower()) != None:
            tk = 'T_TAG_PHP_END'
            # tk = 'T_TAG_END'
            FEATURE_LIST.append(tk)
        
        # <-- && --> comment tag
        elif re.match(pattern=r' *< *! *-- *', string=token.lower()) != None:
            tk = 'T_TAG_COMMENT_START'
            FEATURE_LIST.append(tk)
        elif re.match(pattern=r' *-- *> *', string=token.lower()) != None:
            tk = 'T_TAG_COMMENT_END'
            FEATURE_LIST.append(tk)
        

        # [Important] Variable assignments && Sanitizer
        elif re.match(pattern=var_assign_pattern, string=token) != None:

            # User input through php super global: $a = $_GET['aaa']
            if len(re.findall(pattern=var_source_pattern, string=token.upper())) != 0:
                finds = re.findall(pattern=r'(\$_[A-Z]+)', string=token.upper())
                glb = str(finds[0]).upper()
                if glb in GLOBALS:
                    tk = 'T_VAR_SOURCE_' + glb
                    FEATURE_LIST.append(tk)
            
            # Variable assignments through functions: $xxx = func(a, b, ...) 
            elif len(re.findall(pattern=var_func_pattern, string=token)) != 0:
                func_block_finds = re.findall(pattern=var_func_pattern, string=token)[0]
                func_name = re.findall(pattern=r'(\S+)\(.*\)', string=func_block_finds)[0]
                params_block = re.findall(pattern=r'\S+\((.*)\)', string=func_block_finds)[0].split(',')

                # Use built-in func to sanitize
                if func_name in BUILTIN_SANITIZE_FUNCS:
                    tk = 'T_VAR_SANITIZE_BY_' + func_name.upper()
                    if len(params_block) != 0:
                        for param in params_block:
                            if '$' not in param:
                                param = re.findall(pattern=r' *(\w+) *', string=param)
                                if len(param) == 0:
                                    continue
                                # print(param)
                                sanitize_method = param[0].upper()
                                tk += '_' + sanitize_method
                    FEATURE_LIST.append(tk)
                else:
                    tk = 'T_VAR_ASSIGN_BY_ORDINARY_FUNCTION'
                    FEATURE_LIST.append(tk)
            
            # Variables type converts
            elif len(re.findall(pattern=r'\((\w+)\)', string=token)) != 0:
                finds = re.findall(pattern=r'\((\w+)\)', string=token)
                conv_type =  str(finds[0]).upper()
                tk = 'T_VAR_CONVERT_TYPE_' + conv_type
                FEATURE_LIST.append(tk)

            else:
                tk = 'T_VAR_ASSIGN_COMMON'
                FEATURE_LIST.append(tk)

        # [Important] Variables outputs && sinks sorted by the previous K-Means Model
        elif 'echo' in token.lower():
            sink = re.findall(pattern=r'echo\s(.+)\s*', string=token)[0]
            sink_type = int(sink_predict_class(sink))
            tk = SINK_TYPE_DICT[sink_type]
            FEATURE_LIST.append(tk)
            

        
        # logic branches
        elif 'if' in token.lower():
            tk = 'T_LOGIC_IF'
            FEATURE_LIST.append(tk)
        elif 'elif' in token.lower():
            tk = 'T_LOGIC_ELIF'
            FEATURE_LIST.append(tk)
        elif 'else' in token.lower():
            tk = 'T_LOGIC_ELSE'
            FEATURE_LIST.append(tk)
        elif 'while' in token.lower() or 'for' in token.lower():
            tk = 'T_LOGIC_LOOP'
            FEATURE_LIST.append(tk)
        elif 'break' in token.lower():
            tk = 'T_LOGIC_BREAK'
            FEATURE_LIST.append(tk)
        elif 'continue' in token.lower():
            tk = 'T_LOGIC_CONTINUE'
            FEATURE_LIST.append(tk)
        else:
            continue
            # tk = 'T_UNKNOWN_STAT'
            # FEATURE_LIST.append(tk)
        
    return FEATURE_LIST
