import importlib
import pdb
from pprint import pprint

def import_class(module='', classname='', package=''):
    module = importlib.import_module(f'{modulename}', package=package)
    return getattr(module, class_name)

if __name__ == "__main__":
    import sys
    sys.path.append('test')
    modulename = "test_import"
    class_name = "ImportTest"
    package = "test"
    test_class = import_class(modulename, class_name, package)    
    pprint(test_class)
