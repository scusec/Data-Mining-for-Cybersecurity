def get_relative_array(visited_order):
    '''
    这个函数用于产生访问路径的相对关系，满足以下关系
    1.第一个访问的关系被标为-1,因为其前面没有访问，无法判别相对关系
    2.访问同一个文件标为0
    3.访问不同文件，标为目录切换次数+1
    :param visited_order: 访问路径的序列，每条路径的格式为"/xx/yy/ss/ddd?aaa=bb&ccc=dd"
    :return: 标记路径相对位置的序列
    '''
    order_list = [-1]
    if len(visited_order) == 0:
        return
    pre = None
    for path in visited_order:
        path = path.split("?")[0]  # 取?前面的内容
        if pre == None:
            pre = path
            continue
        if pre == path:  # 两次访问的文件相同
            order_list.append(0)
        else:  # 文件不相同
            flag = 1
            pre_list = pre.split("/")
            path_list = path.split("/")
            pre = path
            the_longer = len(pre_list) if len(pre_list) > len(path_list) else len(path_list)
            the_shorter = len(pre_list) if len(pre_list) < len(path_list) else len(path_list)
            for i in range(0, the_shorter):
                if pre_list[i] != path_list[i]:  # 逐个对比每一层路径
                    order_list.append(len(pre_list) + len(path_list) - 2 * i - 1)
                    flag = 0
                    break;
            if flag == 1:
                order_list.append(the_longer - the_shorter + 1)
    return order_list


def run(filename):
    '''
    :filename: 输入的csv文件名，文件内用逗号间隔
    '''
    from functools import reduce
    with open(filename, "r") as f_input, open(filename[:-4]+"processed"+filename[-4:], "w") as f_output:
        input_str = f_input.readline()
        while input_str:
            input_array = input_str.split(",")
            result_array = get_relative_array(input_array)
            result_str = reduce(lambda x, y: str(x) + "," + str(y), result_array)
            try:
                f_output.write(result_str + "\n")
            except:
                f_output.write(str(result_str) + "\n")
            input_str = f_input.readline()
