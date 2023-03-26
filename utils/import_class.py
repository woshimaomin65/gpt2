import importlib
import pdb
from pprint import pprint

def import_class(module='', classname='', package='', **kw):
    module_instance = importlib.import_module(f'{package}.{module}')
    return getattr(module_instance, classname)

if __name__ == "__main__":
    import sys
    sys.path.append('test')
    modulename = "test_import"
    class_name = "ImportTest"
    package = "test"
    test_class = import_class(modulename, class_name, package)    
    pprint(test_class)
