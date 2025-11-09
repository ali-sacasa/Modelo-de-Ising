#ifndef SOLVER_DE_HPP
#define SOLVER_DE_HPP

#include <vector>
#include <complex>
#include <memory>
#include <string>

// Alias para tipos complejos
using Complex = std::complex<double>;
using VectorXc = std::vector<Complex>;
using MatrixXc = std::vector<VectorXc>;

// Configuración del sistema
struct SystemConfig {
    int N = 6;                    // Número de espines
    double J = 1.0;               // Acoplamiento
    double h = 0.5;               // Campo transversal
    bool periodic = false;
    
    int S = 200;                  // Puntos de muestreo
    int M_trunc = 30;             // Dimensión truncada
    
    double T = 8.0;               // Tiempo total
    int nt = 201;                 // Puntos temporales
};

// Matrices de Pauli
namespace PauliMatrices {
    extern const MatrixXc sx;
    extern const MatrixXc sy;
    extern const MatrixXc sz;
    extern const MatrixXc id2;
}

// Clase principal del solver
class QuantumIsingSolver {
private:
    SystemConfig config;
    int dim;  // 2^N
    
    // Operadores y matrices
    MatrixXc H_dense;  // Hamiltoniano en forma densa
    MatrixXc P;        // Matriz de proyección
    
public:
    QuantumIsingSolver(const SystemConfig& cfg);
    
    // Construcción del Hamiltoniano
    void buildIsingHamiltonian();
    
    // Estados coherentes
    VectorXc coherentState(double z_real, double z_imag) const;
    MatrixXc buildCoherentVectors() const;
    
    // Proyección
    void buildProjection(const MatrixXc& Data);
    
    // Evolución temporal
    std::vector<MatrixXc> timeEvolution(const VectorXc& psi0, 
                                      const std::vector<double>& t_list) const;
    
    // Operadores de observables
    MatrixXc buildSzTotal() const;
    double computeMagnetization(const VectorXc& state) const;
    
    // Utilidades
    static MatrixXc matrixExponential(const MatrixXc& A, double t);
    static MatrixXc matrixMultiply(const MatrixXc& A, const MatrixXc& B);
    static MatrixXc hermitianConjugate(const MatrixXc& A);
    
    // Getters
    const MatrixXc& getHamiltonian() const { return H_dense; }
    const MatrixXc& getProjection() const { return P; }
    int getDimension() const { return dim; }
};

// Utilidades matemáticas
namespace MathUtils {
    // Producto de Kronecker
    MatrixXc kron(const MatrixXc& A, const MatrixXc& B);
    
    // SVD simplificado
    void svd(const MatrixXc& A, MatrixXc& U, std::vector<double>& S, MatrixXc& Vh);
    
    // Diagonalización
    void eigenDecomposition(const MatrixXc& A, std::vector<double>& eigenvalues, 
                          MatrixXc& eigenvectors);
    
    // Norma y productos internos
    double norm(const VectorXc& v);
    Complex dotProduct(const VectorXc& a, const VectorXc& b);
}

// Visualización (salida de datos para plotting)
namespace Visualizer {
    void saveMatrix(const MatrixXc& mat, const std::string& filename);
    void saveVector(const std::vector<double>& vec, const std::string& filename);
    void saveComplexVector(const VectorXc& vec, const std::string& filename);
}

#endif