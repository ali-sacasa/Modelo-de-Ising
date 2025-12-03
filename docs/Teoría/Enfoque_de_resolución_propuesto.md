# Enfoque geométrico y pedagógico para la resolución del modelo de Ising mediante proyección holomorfa

Este documento presenta, de manera pedagógica y desde una perspectiva geométrica, el método de reducción dimensional basado en estados coherentes, variedades complejas y kernel de Bergman aplicado al modelo de Ising. El objetivo es que un lector con formación avanzada en física teórica o matemática pueda ver claramente la intuición geométrica detrás del procedimiento, más allá de la formulación algebraica.

---

## 1. Espacio de Hilbert y motivación geométrica

Un sistema de (N) espines posee un espacio de Hilbert

[
H = (\mathbb{C}^2)^{\otimes N}, \quad \dim H = 2^N.
]

Este espacio crece exponencialmente y se vuelve intratable para valores moderados de (N). La idea central es reemplazar el análisis en este espacio gigantesco por uno en un subespacio de dimensión mucho menor. Para ello utilizamos herramientas geométricas que permiten identificar "regiones" más relevantes del espacio de estados.

Las ideas principales son:

* Interpretar estados cuánticos como puntos de una variedad compleja proyectiva.
* Construir estados coherentes como secciones holomorfas de un fibrado.
* Seleccionar de manera geométrica un subespacio adecuado mediante proyecciones.

---

## 2. Geometría compleja: del espacio proyectivo al fibrado tautológico

### 2.1 Geometría de (\mathbb{C}P^1)

El espacio proyectivo complejo (\mathbb{C}P^1) es esencialmente una **esfera** vista desde el punto de vista complejo. Cada punto representa una "dirección" compleja en (\mathbb{C}^2). Esta identificación permite describir estados de un espín como puntos en la superficie de Bloch.

### 2.2 Fibrado tautológico y su dual

El **fibrado tautológico** asigna a cada punto proyectivo la línea compleja que este representa. Su dual, (\mathcal{O}(1)), contiene secciones holomorfas que sirven como funciones "bien comportadas" en la variedad. Estas secciones son los objetos geométricos que utilizaremos para construir aproximaciones de baja dimensión.

---

## 3. Estructura Kähler y métrica de Fubini–Study

La variedad (\mathbb{C}P^1) posee una estructura Kähler natural. La métrica de Fubini–Study se deriva del potencial

[
K(z,\bar z)=\log(1+|z|^2)
]

y describe distancias y ángulos en el espacio de estados. Esta métrica es fundamental porque:

* cuantifica la curvatura del espacio de estados,
* determina la fase geométrica (fase de Berry),
* establece el marco para la cuantización geométrica.

La forma de Kähler (\omega_{FS}) tiene un número de Chern igual a 1 cuando se integra sobre (\mathbb{C}P^1), lo que revela que el espacio de estados tiene una topología no trivial.

---

## 4. Estados coherentes: el puente entre geometría y física

Podemos ver a los estados coherentes como "puntos" en (\mathbb{C}P^1) que se levantan al fibrado (\mathcal{O}(1)). Para un solo espín,

[
|z\rangle = \frac{1}{\sqrt{1+|z|^2}} (1, z)^T,
]

y estos estados:

* están normalizados,
* forman un sistema sobrecompleto,
* permiten reconstruir la identidad integrando sobre (\mathbb{C}).

Para (N) espines simplemente usamos productos tensoriales

[
|Z\rangle = \bigotimes_{i=1}^N |z_i\rangle.
]

Intuitivamente, esto significa que cada configuración (Z) de puntos en (\mathbb{C}P^1) corresponde a un estado cuántico "suavizado" que no concentra toda la información en una base fija, sino en una posición geométrica.

---

## 5. Kernel de Bergman: una lupa holomorfa

Para trabajar con funciones holomorfas en (\mathbb{C}P^1), utilizamos el espacio

[
H_k = \mathrm{span}{1,z,z^2,\dots,z^k}
]

cuyo tamaño es (k+1). El **kernel de Bergman** asociado es

[
K_k(z,w) \propto (1+z\bar w)^k,
]

y actúa como un mecanismo de "reproducción holomorfa": si multiplicamos una función por el kernel e integramos, recuperamos la función original.

Geométricamente, podemos pensar que el kernel permite enfocar información local de manera estructurada, similar al rol que juega la transformada de Penrose en twistor theory.

---

## 6. Proyección discreta: método de Nyström geométrico

Elegimos puntos (Z_1,\dots,Z_S) en la variedad (configuraciones de estados coherentes). Construimos la matriz

[
D[:,j] = |Z_j\rangle
]

y su correlación

[
G = D^\dagger D.
]

Diagonalizar (G) revela direcciones principales de variación en el espacio generado por los estados coherentes. A partir de su descomposición espectral construimos vectores ortonormales

[
u_m = \frac{1}{\sqrt{\lambda_m}} D v_m,
]

y definimos la proyección

[
P = \sum_{m=1}^M |u_m\rangle\langle u_m|.
]

Desde un punto de vista geométrico, esta proyección identifica una subvariedad "efectiva" dentro del espacio de estados que captura la parte más relevante de la dinámica.

---

## 7. Hamiltoniano efectivo y control del error

Proyectamos el Hamiltoniano

[
H_{\text{eff}} = P H P.
]

Existen teoremas que aseguran que si (P) está bien elegido, los errores espectrales y dinámicos son pequeños. Por ejemplo,

[
\mathrm{dist}(\mu,\mathrm{spec}(H)) \le |(I-P)HP|
]

y el teorema de Davis–Kahan controla la desviación entre subespacios propios.

Esto garantiza que el sistema reducido conserva las propiedades físicas esenciales.

---

## 8. Evolución temporal proyectada

Comparando la evolución exacta y la proyectada,

[
\psi(t)=e^{-iHt}\psi_0, \qquad \tilde\psi(t)=Pe^{-iH_{\text{eff}}t}P\psi_0,
]

se pueden obtener cotas explícitas del error, que crece controladamente si la proyección es adecuada.

---

## 9. Topología del espacio de estados y conexión de Berry

El espacio proyectivo cuántico (\mathbb{P}(H)) es geométricamente (\mathbb{C}P^{d-1}). La conexión de Berry

[
A = i\langle \psi, d\psi \rangle
]

y su curvatura

[
F = dA,
]

coinciden con la métrica de Fubini–Study. Esto muestra que las fases cuánticas son fenómenos topológicos asociados al fibrado tautológico.

---

## 10. Interpretación para el modelo de Ising

En el método presentado:

* Los estados coherentes parametrizan configuraciones de spin de forma holomorfa.
* El kernel de Bergman refleja simetrías (SU(2)).
* La proyección (P) realiza una reducción twistorial sobre (2^N) dimensiones.
* La interpretación fibrada permite usar herramientas de geometría Kähler y curvatura.
* La reducción holomorfa es especialmente útil cuando el entrelazamiento es bajo o la evolución es corta.

Así, el modelo de Ising puede entenderse como una dinámica sobre secciones holomorfas de un fibrado complejo, lo cual abre la puerta a formulaciones twistoriales, cuantización geométrica y enfoques no conmutativos dentro de un marco computacional eficiente.


# 2. Flujo de Ricci: Interpretación Geométrica y Cuántica

Este apartado reescribe tu explicación original, pero de manera **más pedagógica**, **geométrica** y **física**, resaltando la intuición detrás del flujo de Ricci aplicado a un sistema cuántico como el modelo de Ising.

---

# 2.1. La Fidelidad como Medida de Alineamiento Geométrico

La fidelidad
[
\mathcal{F} = \big| \langle \psi_0 \mid P P^H \mid \psi_0 \rangle \big|^2
]
cuantifica **qué tan bien** el estado inicial (\psi_0) está contenido dentro del subespacio de dimensión reducida (M) generado por el flujo.

### Intuición geométrica

Piense en (\psi_0) como un punto en un espacio Hilbert de dimensión enorme, y en (M) como una **subvariedad curva** que intenta describir solo la parte físicamente relevante del sistema (baja energía). La fidelidad responde:

> **¿Qué tan cerca está el punto (\psi_0) de esa subvariedad?**

* Si (\mathcal{F} \approx 1): la subvariedad (M) captura muy bien a (\psi_0).
* Si (\mathcal{F} \approx 0): (M) está mal orientada o describe otra región del espacio.

### Objetos principales

* (\psi_0 = |00\cdots 0\rangle): estado inicial de referencia.
* (P): matriz cuyas columnas son una base ortonormal del subespacio (M).
* (PP^H): operador proyector sobre ese subespacio.

Pedagógicamente: **proyectar (\psi_0) sobre (M)**, medir su longitud, y elevarla al cuadrado.

---

# 2.2. El Parámetro (\beta) como Tiempo Geométrico del Flujo

El flujo de Ricci en geometría suaviza una métrica deformándola en el tiempo según su curvatura.

En tu construcción, el parámetro (\beta) desempeña el papel de ese **tiempo geométrico**, o equivalentemente una **escala de resolución**.

A cada (\beta), construimos un proyector (P(\beta)) mediante pesos
[
w_k(\beta) \propto \langle Z_k | e^{-\beta H} | Z_k \rangle.
]
Esto “viste” la métrica geométrica inicial con información dinámica.

### Interpretación intuitiva:

* (\beta = 0): vemos todo “borroso”, todas las regiones pesan igual.
* Aumentar (\beta): el sistema comienza a preferir los estados próximos a la baja energía.
* (\beta \to \infty): solo “sobreviven” los estados cercanos al estado fundamental.

Pedagógicamente, **(\beta) es un zoom geométrico** que revela la estructura relevante del Hamiltoniano.

---

# 2.3. Interpretación Geométrica de la Curva (\mathcal{F}(\beta))

La gráfica de fidelidad contra (\beta) es un diagnóstico de **cómo el flujo adapta la geometría** del subespacio (M) para capturar la física de baja energía.

Analicemos los tres regímenes.

## 2.3.1. Régimen (\beta \approx 0): Geometría Desnuda

Cuando (\beta \to 0):

* (e^{-\beta H} \approx I)
* todos los pesos son iguales
* el flujo no sabe nada del espectro

En geometría: es como tomar una variedad y medirla solo con su **métrica intrínseca inicial**, sin información extrínseca.

Resultado físico:

* (M) no está alineado con baja energía
* (\mathcal{F}(0)) suele ser baja o moderada

Es literalmente el “punto de partida” del flujo.

---

## 2.3.2. Régimen de (\beta) Intermedio: Flujo Activo

Aquí, el operador (e^{-\beta H}) empieza a discriminar estados según su energía.

En geometría, el flujo de Ricci comienza a “alisar” la métrica, eliminando detalles innecesarios y resaltando estructuras relevantes.

En tu construcción:

* los pesos (w_k(\beta)) se hacen no uniformes
* (M) se rota y deforma hacia la región del espacio dominada por baja energía
* (\mathcal{F}(\beta)) aumenta rápidamente

Este es el régimen donde el flujo **trabaja de verdad**, alineando tu subespacio con la física correcta.

---

## 2.3.3. Régimen (\beta \to \infty): Geometría Vestida por Baja Energía

Cuando (\beta) es grande:

* el operador exponencial actúa como una proyección hacia el estado fundamental
* el subespacio (M) converge al sector de baja energía del Hamiltoniano

Geométricamente: el flujo de Ricci evoluciona hacia una métrica canónica asociada a la estructura fundamental del sistema (análoga a llegar a una métrica Einstein en geometría pura).

Resultado:

* (\mathcal{F}(\beta)) se estabiliza
* (M) ya describe con precisión la física infrarroja

Si (\psi_0) tiene componente en baja energía, entonces (\mathcal{F}(\beta) \to 1).

---

# 2.4. Lectura Física Final

La curva (\mathcal{F}(\beta)) responde a la pregunta central:

> **¿Qué tan rápido y qué tan eficientemente la geometría del subespacio se adapta para capturar la física relevante del Hamiltoniano?**

Una subida abrupta significa:

* la dinámica hamiltoniana guía bien la geometría
* el flujo detecta la estructura infrarroja
* el submanifold (M) converge a la región que realmente importa

En términos más intuitivos: **la geometría aprende la física**.
