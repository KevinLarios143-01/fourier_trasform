from django.shortcuts import render

# Create your views here.
import numpy as np
from .fourier import fourier_series_truncated

from scipy.integrate import quad

# Lista de funciones permitidas
funciones_disponibles = {
    'sin(t)': np.sin,
    'cos(t)': np.cos,
    't^2': lambda t: t**2,
    't': lambda t: t,
    '1': lambda t: np.ones_like(t),
    '0': lambda t: np.zeros_like(t)
}

def fourier_view(request):
    if request.method == "POST":
        # Obtener el período y el número de términos
        T = float(request.POST.get('period', 2 * np.pi))
        N = int(request.POST.get('terms', 5))

        # Obtener la función ingresada por el usuario
        func_str = request.POST.get('function', 'sin(t)')

        # Validar que la función esté en la lista de funciones permitidas
        if func_str in funciones_disponibles:
            f = funciones_disponibles[func_str]
        else:
            return render(request, 'fourier_app/fourier.html', {
                'error': f"Función no permitida: {func_str}",
                'T': T,
                'N': N,
                'function': func_str
            })

        # Generar los valores de tiempo y calcular la serie truncada
        t_values = np.linspace(0, T, 500)
        f_values = fourier_series_truncated(f, T, N, t_values)

        # Calcular los coeficientes de Fourier
        a0, a_coeffs, b_coeffs = calculate_fourier_coefficients(f, T, N)

        # Crear la ecuación de la serie truncada
        fourier_equation = generate_fourier_equation(a0, a_coeffs, b_coeffs)

        # Pasar los datos a la plantilla para mostrar el gráfico y más
        context = {
            'function': func_str,
            't_values': t_values.tolist(),
            'f_values': f_values.tolist(),
            'N': N,
            'T': T,
            'a0': a0,
            'a_coeffs': a_coeffs,
            'b_coeffs': b_coeffs,
            'fourier_equation': fourier_equation,
        }
        return render(request, 'fourier_app/fourier.html', context)

    return render(request, 'fourier_app/fourier.html')

def calculate_a0(f, T):
    """Calcula el coeficiente a0 de Fourier."""
    integral_result, _ = quad(f, 0, T)
    a0 = (1 / T) * integral_result
    return a0

def calculate_an(f, T, n):
    """Calcula el coeficiente an de Fourier para un valor de n."""
    omega = 2 * np.pi / T
    integrand = lambda t: f(t) * np.cos(n * omega * t)
    integral_result, _ = quad(integrand, 0, T)
    an = (2 / T) * integral_result
    return an

def calculate_bn(f, T, n):
    """Calcula el coeficiente bn de Fourier para un valor de n."""
    omega = 2 * np.pi / T
    integrand = lambda t: f(t) * np.sin(n * omega * t)
    integral_result, _ = quad(integrand, 0, T)
    bn = (2 / T) * integral_result
    return bn

def calculate_fourier_coefficients(f, T, N):
    """Calcula los coeficientes a0, an y bn para la serie de Fourier."""
    # Calcular a0
    a0 = calculate_a0(f, T)
    # Calcular los coeficientes an y bn
    a_coeffs = []
    b_coeffs = []
    for n in range(1, N + 1):
        an = calculate_an(f, T, n)
        bn = calculate_bn(f, T, n)
        a_coeffs.append(an)
        b_coeffs.append(bn)

    return a0, a_coeffs, b_coeffs

def generate_fourier_equation(a0, a_coeffs, b_coeffs):
    """Genera la ecuación de la serie truncada."""
    terms = [f"{a0/2:.2f}"]
    for n, (an, bn) in enumerate(zip(a_coeffs, b_coeffs), start=1):
        terms.append(f"{an:.2f} * cos({n} * ω * t)")
        terms.append(f"{bn:.2f} * sin({n} * ω * t)")
    return " + ".join(terms)
