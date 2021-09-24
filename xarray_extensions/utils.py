
def bind_method(cls, fn):
    name = fn.__name__
    if hasattr(cls, name):
        print("Cannot bind method, method %s already exists" % name)
    else:
        setattr(cls, name, fn)