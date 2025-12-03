# Resolución del Modelo de Ising mediante Proyección Holomorfa y Técnicas Numéricas

Este documento presenta un enfoque físico-cuántico y computacional para simular modelos de Ising de muchos cuerpos mediante proyecciones holomorfas sobre estados coherentes, técnicas de reducción dimensional y aproximaciones numéricas avanzadas. La narrativa sigue la lógica de cómo podemos pasar de un espacio de Hilbert inabordable a un subespacio reducido que capture la dinámica física más relevante, mostrando a la vez la intuición geométrica y las herramientas computacionales que hacen posible esta simplificación.

---

## 1. Espacio de Hilbert y Motivación

Consideremos un sistema de $$N$$ espines. Su espacio de Hilbert está dado por

$$
\mathcal{H} = (\mathbb{C}^2)^{\otimes N}, \quad \dim \mathcal{H} = 2^N,
$$

lo que crece exponencialmente con $$N$$ y hace que el tratamiento directo sea impracticable para sistemas moderadamente grandes. La estrategia consiste en proyectar la dinámica sobre un subespacio mucho más pequeño, que capture los modos de baja energía o los estados de alta fidelidad con la configuración inicial. Para ello representamos los estados cuánticos mediante **estados coherentes** sobre $$\mathbb{C}P^1$$, de modo que cada espín individual se describe como

$$
|z\rangle = \frac{1}{\sqrt{1+|z|^2}} 
\begin{pmatrix} 1 \\ z \end{pmatrix}.
$$

Para $$N$$ espines, el estado global se construye como producto tensorial

$$
|Z\rangle = \bigotimes_{i=1}^N |z_i\rangle,
$$

generando un conjunto continuo de configuraciones que actúan como el “esqueleto” de la dinámica del sistema, capturando la esencia de los estados relevantes.

---

## 2. Proyección Holomorfa y Subespacios Relevantes

Para identificar el subespacio que realmente importa, se construye una **matriz de correlación** a partir de un conjunto de configuraciones $$\{Z_j\}$$:

$$
D[:,j] = |Z_j\rangle, \quad G = D^\dagger D.
$$

La diagonalización de $$G$$ permite encontrar las direcciones principales de variación en el espacio generado por los estados coherentes. Los vectores propios con los mayores valores propios forman una base ortonormal $$u_m$$ del subespacio de dimensión reducida $$M$$, y el proyector correspondiente se define como

$$
P = \sum_{m=1}^M |u_m\rangle\langle u_m|.
$$

Físicamente, esto equivale a filtrar el espacio de Hilbert, conservando solo los componentes más relevantes para la dinámica de baja energía.

---

## 3. Hamiltoniano Efectivo y Evolución Temporal

El Hamiltoniano se proyecta sobre este subespacio:

$$
H_{\mathrm{eff}} = P H P,
$$

y la evolución temporal se aproxima mediante

$$
|\tilde\psi(t)\rangle = P e^{-i H_{\mathrm{eff}} t} P |\psi_0\rangle.
$$

Este procedimiento permite simular la dinámica de sistemas grandes con un costo computacional mucho menor, mientras se controla el error

$$
\|\psi(t) - \tilde\psi(t)\| \le \mathcal{O}(\|(I-P) H P\| t),
$$

garantizando que la proyección conserva la mayor parte de la información física relevante.

---

## 4. Flujo de Ricci y Tiempo Geométrico

Se introduce un parámetro $$\beta$$ que actúa como **tiempo imaginario** para guiar la proyección hacia estados de baja energía:

$$
w_k(\beta) \propto \langle Z_k | e^{-\beta H} | Z_k \rangle.
$$

Para $$\beta = 0$$, todos los estados pesan igual; a medida que $$\beta$$ aumenta, la proyección privilegia los estados cercanos al estado fundamental, deformando el subespacio $$M$$ para alinearlo con la dinámica relevante. La **fidelidad**

$$
\mathcal{F}(\beta) = \big| \langle \psi_0 | P(\beta) P(\beta)^\dagger | \psi_0 \rangle \big|^2
$$

cuantifica la calidad de esta alineación, y su evolución con $$\beta$$ proporciona información sobre la eficiencia del flujo en capturar la física infrarroja del sistema.

---

## 5. Análisis de Energía y Control de Error

El Hamiltoniano proyectado permite calcular fácilmente el **promedio de energía** y su dispersión:

$$
\langle H \rangle_\beta = \langle \tilde\psi(\beta) | H | \tilde\psi(\beta)\rangle, \quad
\Delta E_\beta^2 = \langle H^2 \rangle_\beta - \langle H \rangle_\beta^2.
$$

El comportamiento de $$\langle H \rangle_\beta$$ y $$\Delta E_\beta$$ permite evaluar la fidelidad de la proyección y asegurar que se capturan los modos más relevantes. Una dispersión pequeña indica que el subespacio reducido describe con precisión la física de baja energía.

---

## 6. Métodos Computacionales

Para implementar la evolución temporal de manera eficiente se utilizan técnicas numéricas avanzadas:

**Polinomios de Chebyshev:** Permiten aproximar la acción del operador exponencial $$e^{-i H t}$$ sin diagonalizar completamente el Hamiltoniano. La expansión en Chebyshev

$$
e^{-i H t} \approx \sum_{n=0}^{N_c} a_n(t) T_n(H')
$$

reduce el costo computacional y mantiene alta precisión, donde $$T_n$$ son los polinomios de Chebyshev y $$H'$$ es el Hamiltoniano escalado.

**Diagonalización parcial:** Para obtener los vectores principales de $$G$$ se realiza diagonalización parcial usando algoritmos iterativos, que permiten trabajar con matrices de gran tamaño de forma eficiente.

**Proyección discreta y métodos tipo Nyström:** La selección de puntos $$Z_j$$ y la construcción de $$G$$ se hace mediante muestreo inteligente, evitando recorrer todo el espacio de estados. Esto permite construir el subespacio $$M$$ de manera rápida y estable.

---

## 7. Interpretación Física

Este enfoque combina física y computación: los estados coherentes parametrizan configuraciones de spin de forma continua, la proyección $$P$$ identifica los modos más relevantes de la dinámica y la evolución proyectada captura la dinámica de baja energía, incluyendo correlaciones importantes. El flujo de Ricci y el parámetro $$\beta$$ actúan como un mecanismo de filtrado cuántico, mientras que el análisis de energía y dispersión proporciona un control explícito del error. En términos intuitivos, la geometría del subespacio “aprende” la física del Hamiltoniano, concentrando la información relevante en un espacio reducido, mucho más fácil de manejar numéricamente.

---

## 8. Conclusión

El método presentado permite simular sistemas de Ising grandes mediante representaciones continuas de estados coherentes, construcción de subespacios relevantes mediante proyección holomorfa, evolución temporal proyectada usando polinomios de Chebyshev y diagonalización parcial, y control de errores mediante análisis de energía y fidelidad. El resultado es un marco físico-cuántico y computacionalmente eficiente, que proporciona una visión clara de cómo emergen las propiedades de baja energía dentro de un espacio de Hilbert enorme.
