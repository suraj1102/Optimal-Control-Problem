import sympy as sp

constants = {
    "g": "self.gravity",
    "m": "self.mass_bob",
    "l": "self.length_rod",
    "M": "self.mass_cart"
}

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

# print(hjb_pde_str)
print(matlab_to_torch_code("(q11*M^2*l^2*x1^2 + q22*M^2*l^2*x2^2 + q33*M^2*l^2*x3^2 + q44*M^2*l^2*x4^2 + 2*q11*M*l^2*m*x1^2*sin(x1)^2 + 2*q22*M*l^2*m*x2^2*sin(x1)^2 + 2*q33*M*l^2*m*x3^2*sin(x1)^2 + 2*q44*M*l^2*m*x4^2*sin(x1)^2 + q11*l^2*m^2*x1^2*sin(x1)^4 + q22*l^2*m^2*x2^2*sin(x1)^4 + q33*l^2*m^2*x3^2*sin(x1)^4 + q44*l^2*m^2*x4^2*sin(x1)^4)/(l^2*(m*sin(x1)^2 + M)^2) - (cos(x1)^2/(4*l^2*r11*(m*sin(x1)^2 + M)^2))*V_x2^2 - (-cos(x1)/(2*l*r11*(m*sin(x1)^2 + M)^2))*V_x2*V_x4 - ((g*M^2*l*sin(x1) + (sin(2*x1)*M*l^2*m*x2^2)/2 + g*M*l*m*sin(x1)^3 + g*M*l*m*sin(x1) + cos(x1)*l^2*m^2*x2^2*sin(x1)^3 + g*l*m^2*sin(x1)^3)/(l^2*(m*sin(x1)^2 + M)^2))*V_x2 - (-(x4*M^2*l^2 + 2*x4*M*l^2*m*sin(x1)^2 + x4*l^2*m^2*sin(x1)^4)/(l^2*(m*sin(x1)^2 + M)^2))*V_x3 - (1/(4*r11*(m*sin(x1)^2 + M)^2))*V_x4^2 - (-(l^3*m^2*x2^2*sin(x1)^3 + M*l^3*m*x2^2*sin(x1) + g*l^2*m^2*sin(x1)^4 + M*g*l^2*m*sin(x1)^2)/(l^2*(m*sin(x1)^2 + M)^2))*V_x4 - (-(x2*M^2*l^2 + 2*x2*M*l^2*m*sin(x1)^2 + x2*l^2*m^2*sin(x1)^4)/(l^2*(m*sin(x1)^2 + M)^2))*V_x1"))
