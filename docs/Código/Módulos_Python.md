# Módulos en Python (notebooks)

 En Python se implementa el solver completo: Hamiltoniano, flujo de Ricci, evolución con Chebyshev y visualización, lo que permite iterar rápidamente y validar frente a la versión en C++.

## Notebooks incluidos

### `Solver_D.E.s.ipynb`
- Solver principal en Python con álgebra dispersa (`scipy.sparse`), proyección holomorfa y flujo de Ricci (SVD aleatoria vía `sklearn.utils.extmath.randomized_svd`), y propagación con polinomios de Chebyshev (`scipy.special`).
- Permite ajustar parámetros (`N`, `J`, `h`, `S`, `M_trunc`, `T`, `nt`, `cheb_order`, `BETA_MAX`, `n_beta`) y graficar energía, magnetización y fidelidad. Si `quimb` está instalado, se pueden validar espectros.

### `análisis_datos_generados_en_c++.ipynb`
- Lee los CSV producidos por cualquiera de las ejecuciones (C++ o Python si exportas datos equivalentes).
- Genera gráficas de energía total y descomposición ZZ/X, magnetización, varianza y fidelidad a lo largo del tiempo.

### `modelo_analítico.ipynb`
- Apuntes y cálculos de referencia (soluciones exactas en tamaños pequeños o en 1D) para comparar con las simulaciones numéricas.
- Útil para verificar tendencias o valores críticos cuando se modifican parámetros como $J$, $h$ o $N$.

## Flujo de uso sugerido
1. Abre `Solver_D.E.s.ipynb`, ejecuta todas las celdas y ajusta los parámetros deseados.
2. Exporta resultados a CSV si quieres compararlos directamente con la versión en C++ o procesarlos fuera de Jupyter.
3. Usa `análisis_datos_generados_en_c++.ipynb` para graficar cualquier corrida y `modelo_analítico.ipynb` para contrastar con referencias teóricas.

