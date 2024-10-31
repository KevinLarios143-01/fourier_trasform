import numpy as np

from scipy.integrate import quad
import numpy as np

def fourier_series_truncated(f, T, N, t_values):
    """
    Calcula la serie de Fourier truncada hasta 2N + 1 términos para una función periódica f,
    utilizando la función quad para la integración numérica.
    
    :param f: Función periódica f(t)
    :param T: Periodo de la función
    :param N: Número de armónicos truncados
    :param t_values: Valores de t para evaluar la serie de Fourier
    :return: Valores de la serie truncada evaluada en t_values
    """
    omega = 2 * np.pi / T  # Frecuencia angular

    # Cálculo del coeficiente a0
    a0, _ = quad(f, 0, T)
    a0 = (2 / T) * a0
    
    # Inicia la serie con el coeficiente a0/2
    series = np.ones_like(t_values) * a0 / 2

    # Funciones integrandas para an y bn
    def an_integrand(t, n, omega, f):
        return f(t) * np.cos(n * omega * t)

    def bn_integrand(t, n, omega, f):
        return f(t) * np.sin(n * omega * t)

    # Agregar los términos de coseno y seno
    for n in range(1, N + 1):
        # Cálculo de an y bn usando quad
        an, _ = quad(an_integrand, 0, T, args=(n, omega, f))
        bn, _ = quad(bn_integrand, 0, T, args=(n, omega, f))
        
        # Multiplicar por 2 / T
        an = (2 / T) * an
        bn = (2 / T) * bn
        
        # Sumar los términos correspondientes a la serie
        series += an * np.cos(n * omega * t_values) + bn * np.sin(n * omega * t_values)

    return series

