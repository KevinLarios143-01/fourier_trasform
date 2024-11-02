from django.shortcuts import render
import numpy as np
from .fourier import fourier_series_truncated, fourier_series_truncated_teorico
from scipy.integrate import quad
import re

# Definimos las funciones disponibles
import re

# Diccionario de funciones y sus equivalentes en Python
funciones_equivalentes = {
    'sin(t)': 'np.sin(t)',
    'cos(t)': 'np.cos(t)',
    't^2': 't**2',
    't': 't',
    '1': 'np.ones_like(t)',
    '0': 'np.zeros_like(t)'
}

def generar_lambda(expresion):
    # Reemplazar cada función en la expresión con su equivalente en Python
    for nombre_funcion, codigo_equivalente in funciones_equivalentes.items():
        expresion = expresion.replace(nombre_funcion, codigo_equivalente)
    
    # Reemplazar cualquier número constante (entero o decimal) por su equivalente en numpy
    # Esto evita la necesidad de incluir cada número en el diccionario
    expresion = re.sub(r'(?<![\w.])(\d+(\.\d+)?)(?![\w.])', r'\1 * np.ones_like(t)', expresion)

    # Crear la función lambda a partir de la expresión generada
    try:
        funcion_lambda = eval(f"lambda t: {expresion}")
    except Exception as e:
        print(f"Error al generar la función lambda: {e}")
        return False
    
    return funcion_lambda

def extraer_intervalo(cadena_intervalo):
    # Usar una expresión regular para encontrar los límites en el formato 'a < t < b'
    coincidencias = re.search(r'(-?\d+(?:\.\d+)?)\s*<\s*t\s*<\s*(-?\d+(?:\.\d+)?)', cadena_intervalo)
    
    # Verificar si la búsqueda fue exitosa
    if coincidencias:
        # Extraer los grupos correspondientes a t_min y t_max y convertirlos a float
        t_min = float(coincidencias.group(1))
        t_max = float(coincidencias.group(2))
        return t_min, t_max
    else:
        # Devuelve None si el formato es incorrecto
        return None

def fourier_view(request):
    if request.method == "POST":
        posibles_N = [3, 5, 10]
        N_3 = posibles_N[0]
        N_5 = posibles_N[1]
        N_10 = posibles_N[2]
        
        func_value1 = request.POST.get('function1', '').strip()
        interval1 = request.POST.get('intervalo1', '').strip()
        
        func_value2 = request.POST.get('function2', '').strip()
        interval2 = request.POST.get('intervalo2', '').strip()
        
        func_value3 = request.POST.get('function3', '').strip()
        interval3 = request.POST.get('intervalo3', '').strip()
        
        # Definir una lista para almacenar las condiciones de la función por partes
         # Lista para almacenar las partes de la función
        partes_funcion = []
        t_max=0
        t_min = 0
        # Generar función lambda y añadir a la lista solo si se proporcionan ambos valores (función e intervalo)
        if func_value1 and interval1:
            funcion_lambda1 = generar_lambda(func_value1)
            start1, end1 = extraer_intervalo(interval1)
            if start1 < t_min:
                t_min = start1
            if end1 > t_max:
                t_max = end1
            if funcion_lambda1:
                partes_funcion.append((funcion_lambda1, start1, end1))

        if func_value2 and interval2:
            funcion_lambda2 = generar_lambda(func_value2)
            start2, end2 = extraer_intervalo(interval2)
            if start2 < t_min:
                t_min = start2
            if end2 > t_max:
                t_max = end2
            if funcion_lambda2:
                partes_funcion.append((funcion_lambda2, start2, end2))

        if func_value3 and interval3:
            funcion_lambda3 = generar_lambda(func_value3)
            start3, end3 = extraer_intervalo(interval3)
            if start3 < t_min:
                t_min = start3
            if end3 > t_max:
                t_max = end3
            if funcion_lambda3:
                partes_funcion.append((funcion_lambda3, start3, end3))

        # Definir la función por partes
        def funcion_por_partes(t):
            for funcion, start, end in partes_funcion:
                if start <= t <= end:
                    return funcion(t)
            return 0  # Valor por defecto si t no está en ningún intervalo
        
        
        T = t_max - t_min            

        # Generar valores de tiempo
        t_values = np.linspace(t_min, t_max, 1000)

        # Evaluar la función original en estos valores de tiempo
        original_values = [float(funcion_por_partes(t)) for t in t_values]

        # Calcular la serie de Fourier truncada (aproximación) en los mismos valores de tiempo para N=3
        f_values_n_3 = fourier_series_truncated(funcion_por_partes, T, N_3, t_values)
        f_values_teorico_n_3 = fourier_series_truncated_teorico(funcion_por_partes, T, N_3, t_values)
        # Calcular la serie de Fourier truncada (aproximación) en los mismos valores de tiempo para N=5
        f_values_n_5 = fourier_series_truncated(funcion_por_partes, T, N_5, t_values)
        f_values_teorico_n_5 = fourier_series_truncated_teorico(funcion_por_partes, T, N_5, t_values)
        # Calcular la serie de Fourier truncada (aproximación) en los mismos valores de tiempo para N=10
        f_values_n_10 = fourier_series_truncated(funcion_por_partes, T, N_10, t_values)
        f_values_teorico_n_10 = fourier_series_truncated_teorico(funcion_por_partes, T, N_10, t_values)
        # Calcular los coeficientes de Fourier
        
       
        a0_3, a_coeffs_3, b_coeffs_3 = calculate_fourier_coefficients(funcion_por_partes, T, N_3)
        a0_5, a_coeffs_5, b_coeffs_5 = calculate_fourier_coefficients(funcion_por_partes, T, N_5)
        a0_10, a_coeffs_10, b_coeffs_10 = calculate_fourier_coefficients(funcion_por_partes, T, N_10)
        
        omegin = 2 * np.pi / T
        
        fourier_equation_3 = generate_fourier_equation(a0_3, a_coeffs_3, b_coeffs_3, omegin)
        fourier_equation_5 = generate_fourier_equation(a0_5, a_coeffs_5, b_coeffs_5, omegin)
        fourier_equation_10 = generate_fourier_equation(a0_10, a_coeffs_10, b_coeffs_10, omegin)

        # Pasar los datos como listas al contexto para el frontend
        context = {
            'T': T,
            'N_3': N_3,
            'a0_3': a0_3,
            'an_3': a_coeffs_3,
            'bn_3': b_coeffs_3,
            'N_5': N_5,
            'a0_5': a0_5,
            'an_5': a_coeffs_5,
            'bn_5': b_coeffs_5,
            'N_10': N_10,
            'a0_10': a0_10,
            'an_10': a_coeffs_10,
            'bn_10': b_coeffs_10,
            't_values': t_values.tolist(),  # Convertimos t_values a lista para JSON
            'original_values': original_values,
            'f_values_3': f_values_n_3.tolist(),  # Convertimos f_values a lista para JSON
            'f_values_teorico_3': f_values_teorico_n_3.tolist(),  # Convertimos f_values a lista para JSON
            'f_values_5': f_values_n_5.tolist(),  # Convertimos f_values a lista para JSON
            'f_values_teorico_5': f_values_teorico_n_5.tolist(),  # Convertimos f_values a lista para JSON
            'f_values_10': f_values_n_10.tolist(),  # Convertimos f_values a lista para JSON
            'f_values_teorico_10': f_values_teorico_n_10.tolist(),  # Convertimos f_values a lista para JSON
            'fourier_equation_3': fourier_equation_3,
            'fourier_equation_5': fourier_equation_5,
            'fourier_equation_10': fourier_equation_10
        }
        return render(request, 'fourier_app/fourier.html', context)

    return render(request, 'fourier_app/fourier.html')

# Funciones de cálculo de coeficientes de Fourier

def calculate_a0(f, T):
    integral_result, _ = quad(f, -T/2, T/2)
    return (1 / T) * integral_result

def calculate_an(f, T, n):
    omega = 2 * np.pi / T
    integrand = lambda t: f(t) * np.cos(n * omega * t)
    integral_result, _ = quad(integrand, -T/2, T/2)
    return (2 / T) * integral_result

def calculate_bn(f, T, n):
    omega = 2 * np.pi / T
    integrand = lambda t: f(t) * np.sin(n * omega * t)
    integral_result, _ = quad(integrand, -T/2, T/2)
    return (2 / T) * integral_result

def calculate_fourier_coefficients(f, T, N):
    # Calcular y redondear a0 a 2 decimales 
    a0 = round(calculate_a0(f, T), 2)
    
    # Calcular y redondear cada an y bn a 2 decimales
    a_coeffs = [round(calculate_an(f, T, n), 2) for n in range(1, N + 1)]
    b_coeffs = [round(calculate_bn(f, T, n), 2) for n in range(1, N + 1)]
    
    return a0, a_coeffs, b_coeffs

def generate_fourier_equation(a0, a_coeffs, b_coeffs, omega):
    # Calcula omega

    # Término constante a0
    terms = [f"{a0:.2f}"]
    
    # Términos de la serie con cosenos y senos (en corchetes)
    series_terms = []
    for n, (an, bn) in enumerate(zip(a_coeffs, b_coeffs), start=1):
        series_terms.append(f"({an:.2f} * cos({n} * {omega:.2f} * t)")
        series_terms.append(f"{bn:.2f} * sin({n} * {omega:.2f} * t))")
    
    # Unir los términos dentro de los corchetes y agregar a a0
    return f"{terms[0]} + [ " + " + ".join(series_terms) + " ]"