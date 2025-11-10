import numpy as np

#matrices de pauli para un spin:

pauliX = np.array([[0,1],
                   [1,0]])

pauliY = np.array([[0, -1j],
                   [1j, 0]])

pauliZ = np.array([[1,0],[0,-1]])


ideMatrix = np.eye(2)


def hamiltoniano(N, J, g, PZ, PX, I):
    """
    
    parámetros 
    N = cantidad de spines
    J = acoplamiento
    g = parámetro energético del campo transversal 
    PZ = la matriz de pauli en la componente Z
    PX = la matriz de pauli en la componente X
    I = matriz identidad
    
    
    """
    dim = 2**N
    
    H = np.zeros((dim, dim))
    
    ## El término ZZ (entre vecinos)
    for i in range (N-1): #por pares, por eso hasta N-1, el siguiente N del último es el primer término 
        
        #necesito un elemento neutro: matriz 1x1
        eleVeci = 1 
        
        #lo que voy a hacer es ir multiplicando cada término en su posición
        
        #en cada suma, un producto:
        
        for j in range(N): #tengo que hacerlo N veces, pues es para posicionar las matrices z entre los N spines
            if j == i:
                eleVeci = np.kron(eleVeci, PZ) #si está en i, poner la matriz z
            elif j==i+1:
                eleVeci = np.kron(eleVeci,PZ) #si está en i+1 también
            else:
                eleVeci = np.kron(eleVeci,I) #sino poner la identidad del espacio
        H += -J*eleVeci 
    
    
    #El término trasversal 
    
    for i in range (N): #ya no es entre pares, sino hasta N
        eleTras = 1
        for j in range(N): #mismo posicionamiento 
            if j == i:
                eleTras = np.kron(eleTras, PX) #misma lógica pero solo para un término sin considerar el siguiente
            else:
                eleTras = np.kron(eleTras, I)
        H += -g*eleTras   
    
    if H.shape == (dim, dim):
        print("yei :)")
    

    return H


hamiltoniano(3,1,1,pauliZ,pauliX, ideMatrix)