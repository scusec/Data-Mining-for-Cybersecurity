import re
import subprocess
import os
from tqdm import tqdm


def php_to_opcode(phpfilename):
    try:
        # 执行指定的命令，如果执行状态码为0则返回命令执行结果，否则抛出异常。
        output = subprocess.check_output(
            ['php', '-dvld.active=1', '-dvld.execute=0', phpfilename], stderr=subprocess.STDOUT)
        output = str(output, encoding='utf-8')
        tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)
        t = " ".join(tokens)
        t = t[6:]
        return t
    except:
        print(
            "\n[-]Warning: something happend when execute vld to get opcode:" + str(phpfilename))
        return " "


def trans(path):
    if os.path.exists(path) == False:
        print('\n[-]Warning: path does not exist, do nothing')
        return [], 0, 0

    result = []
    i = 0
    j = 0
    if os.path.isfile(path):
        result.append(php_to_opcode(path))

    # is a dir
    elif os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for single_file in tqdm(filenames, desc=dirpath, ncols=100):
                if '.php' in single_file:
                    fullpath = os.path.join(dirpath, single_file)
                    single_result = php_to_opcode(fullpath)
                    if single_result == " ":
                        j += 1
                        continue
                    result.append(single_result)
                    i += 1

    return result, i, j
