# Módulos en Python
## Notebooks incluidos

### `Solver_D.E.s.ipynb`
- Inclyye un prototipo del solver con álgebra dispersa (`scipy.sparse`), evolución con polinomios de Chebyshev (`scipy.special`) y muestreo de estados coherentes.
- Incluye la construcción del Hamiltoniano de Ising y cálculo de proyecciones mediante SVD aleatoria (`sklearn.utils.extmath.randomized_svd`)

### `análisis_datos_generados_en_c++.ipynb`
- Lee los CSV producidos por el binario C++ (`energy_results.csv`, `magnetization.csv`, etc.).
- Genera gráficas de energía total y descomposición ZZ/X, magnetización, varianza y fidelidad a lo largo del tiempo.

### `modelo_analítico.ipynb`
- Apuntes y cálculos de referencia, como soluciones exactas en tamaños pequeños o en 1D para comparar con las simulaciones numéricas.
- Funciona para verificar tendencias o valores críticos cuando se modifican parámetros como $$J$$, $$h$$ o $$N$$.


