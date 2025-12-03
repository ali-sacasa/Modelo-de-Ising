# Discusión

## Calidad de la proyección
- La fidelidad inicial depende del número de estados coherentes `S` y de la dimensión truncada `M_trunc`. Aumentar `S` mejora la cobertura de la esfera de Bloch, pero encarece la diagonalización de la matriz de Gram.
- El barrido en $\beta$ actúa como filtro: valores pequeños mantienen diversidad de estados, valores grandes privilegian bajas energías. `ricci_fidelities.csv` permite elegir manualmente otro punto si el óptimo se desea sesgar hacia más/menos energía.

## Errores numéricos
- Los polinomios de Chebyshev requieren que el reescalado del espectro sea estable. El código estima los extremos espectrales diagonalizando el Hamiltoniano denso (válido para tamaños pequeños); para sistemas mayores conviene sustituir por un método iterativo (Lanczos) para mejorar precisión y evitar gasto de memoria.
- La varianza de energía (`variance.csv`) y el error de energía/magnetización (`metrics.csv`) son indicadores directos del ajuste. Si crecen con el tiempo, incrementa `cheb_order` o `M_trunc`.
- Las operaciones densas crecen como $2^N$; para valores de $N$ grandes, el uso de matrices densas en la diagonalización de pesos térmicos puede ser el cuello de botella.

## Interpretación física
- El término transversal $h \sum \sigma^x$ induce fluctuaciones cuánticas que erosionan la magnetización alineada. El seguimiento de `magnetization.csv` permite ver cómo la proyección captura esa pérdida de orden.
- La descomposición de energía ZZ/X diferencia la contribución de correlaciones de pares vs. el campo transversal. Cambios de signo o cruces en estas curvas pueden señalar regímenes dominados por campo vs. interacciones.
- La densidad de energía local puede mostrar la nucleación de dominios o excitaciones localizadas; comparar las matrices completa y proyectada ayuda a evaluar cuánto detalle espacial se preserva.

## Limitaciones actuales
- El programa está pensado para tamaños pequeños/medios (p. ej. $N \leq 10$) debido al coste de las matrices densas en el flujo de Ricci.
- No se incluye paralelismo por defecto; ver la sección de implementación paralela para ideas de escalabilidad.
- El directorio de salida es el directorio de trabajo; si se integran más experimentos conviene parametrizarlo para evitar sobrescrituras.
