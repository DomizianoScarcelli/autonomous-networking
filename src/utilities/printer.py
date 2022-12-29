DEBUG = True
OUT = True

def colored(r, g, b, text):
    if OUT: return text
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_debug_colored(r=0,g=0,b=0, text=""):
    if DEBUG:
        print(colored(r,g,b, text))