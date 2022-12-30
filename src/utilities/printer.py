from .tester import REDICRECT_STDOUT

def colored(r, g, b, text):
    if REDICRECT_STDOUT: return text
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_debug_colored(r=0,g=0,b=0, text=""):
    from .config import TESTER_DEBUG #Just in time import to avoid cicular import
    if TESTER_DEBUG:
        print(colored(r,g,b, text))