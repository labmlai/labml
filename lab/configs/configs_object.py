import ast
import inspect
import warnings


class Configs:
    batch_size: int
    x: ast.NodeTransformer


def dep():
    warnings.warn("inspect.getargspec() is deprecated since Python 3.0, "
                  "use inspect.signature() or inspect.getfullargspec()",
                  DeprecationWarning, stacklevel=2)

    return None


def model(configs: Configs):
    print(2)

    def temp():
        print('x')

    temp()
    z = max(2, configs.batch_size * 2)

    return z, configs, configs.x.visit()


x = inspect.signature(model)
y = inspect.getfullargspec(model)
print(x)

source = inspect.getsource(model)


def visit_Attribute(node: ast.Attribute):
    print('Att', node)
    return False


# Only visits if not captured before
def visit_Name(node: ast.Name):
    print('Name', node)


node_iter = ast.NodeVisitor()
node_iter.visit_Attribute = visit_Attribute
node_iter.visit_Name = visit_Name
parsed = ast.parse(source)
node_iter.visit(parsed)
print(2)
