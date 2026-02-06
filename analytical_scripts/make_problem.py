import sympy as sp

""""CONSTANTS"""
x_dim = 2
u_dim = 1

f_x = [
    ["x2"],
    ["g/l * sin(x1)"]
]

g_x = [
    ["0"],
    ["1/(m*l*l)"]
]

constants = {
    "g": "self.gravity",
    "m": "self.mass",
    "l": "self.length",
    "gamma": "self.gamma",
}
# --------------------------------------------------------------------- #

import sys
assert sys.version_info[:2] == (3, 11), "Python 3.11.x is required."
import matlab.engine

eng = matlab.engine.start_matlab()
eng.eval("clear; clc; close;", nargout=0)

# syms x1 x2 x3 x4 u1 u2 q11 real
x_names = [f"x{i}" for i in range(1, x_dim + 1)]
u_names = [f"u{i}" for i in range(1, u_dim + 1)]
q_entries = [f"q{i}{i}" for i in range(1, x_dim + 1)]
r_entries = [f"r{i}{i}" for i in range(1, u_dim + 1)]
all_vars = " ".join(x_names + u_names + q_entries + r_entries + list(constants.keys()))
eng.eval(f"syms {all_vars} real;", nargout=0)

# x = [x1; x2]; u =  u1;
x_vars_str = "; ".join(x_names)
u_vars_str = "; ".join(u_names)
eng.eval(f"x = [{x_vars_str}];", nargout=0)
eng.eval(f"u = [{u_vars_str}];", nargout=0)

# define the Q and R matrices
q_entries_str = "; ".join(q_entries)
r_entries_str = "; ".join(r_entries)
eng.eval(f"Q = diag([{q_entries_str}]);", nargout=0)
eng.eval(f"R = diag([{r_entries_str}]);", nargout=0)

# define the f and g functions
f_rows = [" ".join(row) for row in f_x]
f_str = "; ".join(f_rows)
g_rows = [" ".join(row) for row in g_x]
g_str = "; ".join(g_rows)
eng.eval(f"f_x = [{f_str}];", nargout=0)
eng.eval(f"g_x = [{g_str}];", nargout=0)

eng.run("derivepdepython.m", nargout=0)

hjb_pde_str = eng.eval("char(HJB);")

def map_symbol(name):
    if name.startswith("q") and len(name) == 3:
        i = int(name[1])
        j = int(name[2])
        return f"Q[{i-1}, {j-1}]"
    
    if name.startswith("r") and len(name) == 3:
        i = int(name[1])
        j = int(name[2])
        return f"R[{i-1}, {j-1}]"
    
    return constants.get(name, name)


def sympy_to_torch(expr):
    if isinstance(expr, sp.Symbol):
        return map_symbol(str(expr))
    
    if isinstance(expr, sp.Number):
        return str(expr)
    
    if isinstance(expr, sp.Add):
        return " + ".join(sympy_to_torch(a) for a in expr.args)
    
    if isinstance(expr, sp.Mul):
        return " * ".join(sympy_to_torch(a) for a in expr.args)
    
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == 2:
                return f"torch.square({sympy_to_torch(base)})"
        return f"{sympy_to_torch(base)} ** {sympy_to_torch(exp)}"
    
    if expr.func == sp.sin:
        return f"torch.sin({sympy_to_torch(expr.args[0])})"
    
    if expr.func == sp.cos:
        return f"torch.cos({sympy_to_torch(expr.args[0])})"
    
    if expr.func == sp.exp:
        return f"torch.exp({sympy_to_torch(expr.args[0])})"
    
    return str(expr)


def matlab_to_torch_code(matlab_expr):
    matlab_expr = matlab_expr.replace("^", "**")
    sym_expr = sp.sympify(matlab_expr)

    if isinstance(sym_expr, sp.Add):
        terms = list(sym_expr.args)
    else:
        terms = [sym_expr]

    lines = []

    for i, term in enumerate(terms):
        torch_term = sympy_to_torch(term)
        lines.append(f"term{i+1} = {torch_term}")

    return_line = " + ".join([f"term{i+1}" for i in range(len(terms))])
    lines.append(f"return {return_line}")

    return "\n".join(lines)

print(hjb_pde_str)
print(matlab_to_torch_code(hjb_pde_str))