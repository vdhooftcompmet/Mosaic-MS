import importlib.util
import sys


class LazyLoader:
    def __init__(self):
        self.interceptor = Interceptor()
        self.modules_to_load = []
        self._initial_sys_modules = None

    def __enter__(self):
        self._initial_sys_modules = set(sys.modules.keys())
        sys.meta_path.insert(0, self.interceptor)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.meta_path.remove(self.interceptor)
        current_sys_modules = set(sys.modules.keys())
        new_modules = current_sys_modules - self._initial_sys_modules
        self.modules_to_load = [sys.modules[mod] for mod in new_modules if sys.modules[mod]]

    def load_all(self):
        for module in self.modules_to_load:
            if hasattr(module, "__spec__"):
                _ = module.__dict__ 


class Interceptor:
    def __init__(self):
        self.processing = set()

    def find_spec(self, fullname, path, target=None):
        return _find_spec(fullname, path, target, self.processing)


def _find_spec(fullname, path, target, processing: set):
    if fullname in processing:
        return None
            
    processing.add(fullname)
    try:
        real_spec = importlib.util.find_spec(fullname, path)
        if real_spec is None:
            return None
        
        real_spec.loader = importlib.util.LazyLoader(real_spec.loader)
        return real_spec
    finally:
        processing.remove(fullname)