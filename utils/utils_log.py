import time

SHOW_SCREEN = True


def log_error(msg: str):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open('error.log', 'a') as f:
        f.write(time_str + ' ' + msg)
        f.write("\n")
        if SHOW_SCREEN:
            print(msg)


def log_info(msg: str):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open('info.log', 'a') as f:
        f.write(time_str + ' ' + msg)
        f.write("\n")
        if SHOW_SCREEN:
            print(msg)


def log_debug(msg: str):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open('debug.log', 'a') as f:
        f.write(time_str + ' ' + msg)
        f.write("\n")
        # if SHOW_SCREEN:
        #     print(msg)


def log_time(msg: str):
    with open('time.log', 'a') as f:
        f.write(msg)
        f.write("\n")
        if SHOW_SCREEN:
            print(msg)
