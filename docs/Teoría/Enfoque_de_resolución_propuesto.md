# Resolución del Modelo de Ising mediante Proyección Holomorfa y Técnicas Numéricas

## 1. Espacio de Hilbert y motivación

Para $$N$$ espines, el espacio de Hilbert es:

$$
\mathcal{H} = (\mathbb{C}^2)^{\otimes N}, \quad \dim \mathcal{H} = 2^N,
$$

Por lo que crece exponencialmente con $$N$$. Se proyecta la dinámica sobre un subespacio más pequeño que capture los modos de baja energía o los estados de alta fidelidad con la configuración inicial. Cada espín se representa con un estado coherente sobre $$\mathbb{C}P^1$$:

$$
|z\rangle = \frac{1}{\sqrt{1+|z|^2}}
\begin{pmatrix} 1 \\ z \end{pmatrix},
$$

Para $$N$$ espines se usa el producto tensorial $$|Z\rangle = \bigotimes_{i=1}^N |z_i\rangle$$.

---

## 2. Proyección Holomorfa y Subespacios Relevantes

Se construye una matriz de correlación a partir de configuraciones $$\{Z_j\}$$:

$$
D[:,j] = |Z_j\rangle, \quad G = D^\dagger D.
$$

La diagonalización de $$G$$ permite hallar las direcciones principales. Los autovectores asociados a los mayores autovalores forman una base ortonormal del subespacio de dimensión reducida $$M$$, con proyector:

$$
P = \sum_{m=1}^M |u_m\rangle\langle u_m|.
$$

Físicamente se está filtrando el espacio de Hilbert, conservando solo los componentes relevantes para la dinámica de baja energía.

---

## 3. Hamiltoniano Efectivo y Evolución Temporal

El Hamiltoniano se proyecta:

$$
H_{\mathrm{eff}} = P H P,
$$

De modo que la evolución aproximada es:

$$
|\tilde\psi(t)\rangle = P e^{-i H_{\mathrm{eff}} t} P |\psi_0\rangle.
$$

Se controla el error mediante:

$$
\|\psi(t) - \tilde\psi(t)\| \le \mathcal{O}(\|(I-P) H P\| t),
$$

Entonces, se asegura que la proyección conserva la mayor parte de la información física relevante.

---

## 4. Flujo de Ricci y Tiempo Geométrico

Se introduce un parámetro $$\beta$$ como tiempo imaginario que favorece los estados de baja energía:

$$
w_k(\beta) \propto \langle Z_k | e^{-\beta H} | Z_k \rangle.
$$

Para $$\beta = 0$$ todos los estados pesan igual. Al aumentar $$\beta$$, la proyección favorece configuraciones cercanas al estado fundamental. La fidelidad se evalúa mediante:

$$
\mathcal{F}(\beta) = \big| \langle \psi_0 | P(\beta) P(\beta)^\dagger | \psi_0 \rangle \big|^2
$$


---

## 5. Análisis de Energía y Control de Error

El Hamiltoniano proyectado permite calcular energía** y dispersión según:

$$
\langle H \rangle_\beta = \langle \tilde\psi(\beta) | H | \tilde\psi(\beta)\rangle, \quad
\Delta E_\beta^2 = \langle H^2 \rangle_\beta - \langle H \rangle_\beta^2.
$$

Una dispersión pequeña indica que el subespacio reducido describe con precisión la física de baja energía.

---

## 6. Métodos Computacionales

- **Polinomios de Chebyshev:** aproximan $$e^{-i H t}$$ sin diagonalizar completamente el Hamiltoniano.
- **Diagonalización parcial o SVD aleatoria:** extrae vectores principales de $$G$$ de manera eficiente.
- **Muestreo tipo lattice de Fibonacci:** elige puntos $$Z_j$$ bien distribuidos en la esfera de Bloch.
- **Ponderación térmica:** introduce pesos $$w_k(\beta)$$ para guiar el subespacio hacia la energía mínima.

---

## 7. Significado físico

Los estados coherentes parametrizan configuraciones de spin de forma continua. El proyector $$P$$ identifica los modos más relevantes y la evolución proyectada captura la dinámica de baja energía. El flujo de Ricci actúa como filtro cuántico, mientras que el análisis de energía y fidelidad controla el error.

En suma, el método empleado combina geometría (proyecciones holomórficas), técnicas numéricas y control energético (o de fidelidad) para simular sistemas de Ising grandes en un subespacio más manejable computacionalmente. Entonces, este permite estudiar correlaciones y magnetización con costo computacional reducido y un marco de error explícito.
