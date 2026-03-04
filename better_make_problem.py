import sympy as sp

constants = {
    "g": "self.gravity",
    "m": "self.mass_bob",
    "l": "self.length_rod",
    "M": "self.mass_total",
    "mc": "self.mass_cart"
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
print(matlab_to_torch_code("q11*x1^2 - (cos(x1)^2/(4*l^2*r11*(2*m + 2*mc - m*cos(x1)^2)^2))*V_x2^2 - (- cos(x1)/(2*l*r11*(m + mc)*(2*m - m*cos(x1)^2 + 2*mc)) - (m*cos(x1)^3)/(2*l*r11*(m + mc)*(2*m + 2*mc - m*cos(x1)^2)^2))*V_x2*V_x4 - (-(l*m*cos(x1)*sin(x1)*x2^2 + g*m*sin(x1) + g*mc*sin(x1))/(l*(2*m - m*cos(x1)^2 + 2*mc)))*V_x2 - (-x4)*V_x3 - (1/(r11*(m + mc)*(2*m - m*cos(x1)^2 + 2*mc)) - 1/(r11*(2*m + 2*mc - m*cos(x1)^2)^2) + (m*cos(x1)^2)/(r11*(m + mc)*(2*m + 2*mc - m*cos(x1)^2)^2))*V_x4^2 - (-(l*m*(x2*sin(x1) + (cos(x1)*(l^2*m^2*x2^2*cos(x1)*sin(x1) - g*l*m*sin(x1)*(m + mc)))/(l^2*m*(2*m - m*cos(x1)^2 + 2*mc))))/(m + mc))*V_x4 - (-x2)*V_x1 + q22*x2^2 + q33*x3^2 + q44*x4^2"))
