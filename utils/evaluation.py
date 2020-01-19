import time


def print_time():
    print ('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))