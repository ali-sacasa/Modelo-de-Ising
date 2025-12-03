# Modelo de Ising

Implementación en Python y C++ de un solver para el modelo de Ising cuántico con proyección holomorfa, aproximación de la evolución mediante polinomios de Chebyshev y análisis de energía y magnetización. La versión en Python sirve para experimentar más rápidamente y validar el método, mientras que la versión en C++ prioriza rendimiento manteniendo la misma lógica física y numérica.

## Código
- **Python**:
  - `Solver_D.E.s.ipynb`: solver en Python: Hamiltoniano, flujo de Ricci, proyección, evolución con Chebyshev y visualización.
  - `modelo_analítico.ipynb`: notas teóricas y comparaciones analíticas.
  - `análisis_datos_generados_en_c++.ipynb`: lectura y graficado de los CSV generados por la versión en C++ o por salidas equivalentes en Python.
- **C++**:
  - `Main_C++_Solver_D.E.s.cpp`: programa que configura parámetros, ejecuta el flujo de Ricci, evoluciona el sistema y guarda CSV.
  - `Cpp_Solver_D.E.s.cpp` y `hpp_Solver_D.E.s.hpp`: clases que contienen: operadores de spin, muestreo de estados coherentes, flujo de Ricci, propagador de Chebyshev, análisis y exportación de resultados.


