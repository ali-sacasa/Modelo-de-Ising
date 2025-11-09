#include "hpp_Solver_D.E.s.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

// Definición de matrices de Pauli
const MatrixXc PauliMatrices::sx = {{0, 1}, {1, 0}};
const MatrixXc PauliMatrices::sy = {{0, Complex(0, -1)}, {Complex(0, 1), 0}};
const MatrixXc PauliMatrices::sz = {{1, 0}, {0, -1}};
const MatrixXc PauliMatrices::id2 = {{1, 0}, {0, 1}};

// Implementación de QuantumIsingSolver
QuantumIsingSolver::QuantumIsingSolver(const SystemConfig& cfg) : config(cfg) {
    dim = 1 << config.N;  // 2^N
    H_dense = MatrixXc(dim, VectorXc(dim, 0.0));
    P = MatrixXc(dim, VectorXc(config.M_trunc, 0.0));
}

void QuantumIsingSolver::buildIsingHamiltonian() {
    std::cout << "Construyendo Hamiltoniano de Ising para N=" << config.N 
              << ", dim=" << dim << std::endl;
    
    // Interacción ZZ
    for (int i = 0; i < config.N - 1 + (config.periodic ? 1 : 0); ++i) {
        int j = (i + 1) % config.N;
        
        // Construir operador σ_z^i ⊗ σ_z^j
        MatrixXc term(dim, VectorXc(dim, 1.0));
        
        for (int k = 0; k < config.N; ++k) {
            MatrixXc op;
            if (k == i || k == j) {
                op = PauliMatrices::sz;
            } else {
                op = PauliMatrices::id2;
            }
            term = MathUtils::kron(term, op);
        }
        
        // Sumar al Hamiltoniano: -J * term
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                H_dense[row][col] -= config.J * term[row][col];
            }
        }
    }
    
    // Campo transversal
    for (int i = 0; i < config.N; ++i) {
        MatrixXc term(dim, VectorXc(dim, 1.0));
        
        for (int k = 0; k < config.N; ++k) {
            MatrixXc op = (k == i) ? PauliMatrices::sx : PauliMatrices::id2;
            term = MathUtils::kron(term, op);
        }
        
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                H_dense[row][col] -= config.h * term[row][col];
            }
        }
    }
    
    std::cout << "Hamiltoniano construido." << std::endl;
}

VectorXc QuantumIsingSolver::coherentState(double z_real, double z_imag) const {
    Complex z(z_real, z_imag);
    double norm = 1.0 / std::sqrt(1.0 + std::norm(z));
    return VectorXc{1.0, z} * norm;
}

MatrixXc QuantumIsingSolver::buildCoherentVectors() const {
    std::cout << "Muestreando " << config.S << " estados coherentes..." << std::endl;
    
    MatrixXc Data(dim, VectorXc(config.S, 0.0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> angle(0.0, 2.0 * M_PI);
    
    for (int j = 0; j < config.S; ++j) {
        // Muestreo uniforme en CP^1
        double u = uniform(gen);
        double phi = angle(gen);
        double theta = std::acos(u);
        Complex z = std::tan(theta / 2.0) * std::exp(Complex(0, phi));
        
        // Construir estado producto |z⟩^{\otimes N}
        VectorXc psi = coherentState(z.real(), z.imag());
        for (int i = 1; i < config.N; ++i) {
            VectorXc single_site = coherentState(z.real(), z.imag());
            VectorXc new_psi(psi.size() * single_site.size());
            for (size_t a = 0; a < psi.size(); ++a) {
                for (size_t b = 0; b < single_site.size(); ++b) {
                    new_psi[a * single_site.size() + b] = psi[a] * single_site[b];
                }
            }
            psi = new_psi;
        }
        
        // Normalizar
        double norm_psi = MathUtils::norm(psi);
        for (size_t i = 0; i < psi.size(); ++i) {
            Data[i][j] = psi[i] / norm_psi;
        }
    }
    
    return Data;
}

void QuantumIsingSolver::buildProjection(const MatrixXc& Data) {
    std::cout << "Construyendo proyección..." << std::endl;
    
    // Matriz Gram G = D^† D
    MatrixXc G(config.S, VectorXc(config.S, 0.0));
    for (int i = 0; i < config.S; ++i) {
        for (int j = 0; j < config.S; ++j) {
            for (int k = 0; k < dim; ++k) {
                G[i][j] += std::conj(Data[k][i]) * Data[k][j];
            }
        }
    }
    
    // Diagonalización
    std::vector<double> eigenvalues;
    MatrixXc eigenvectors;
    MathUtils::eigenDecomposition(G, eigenvalues, eigenvectors);
    
    // Ordenar por valor propio descendente
    std::vector<size_t> indices(eigenvalues.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), 
              [&](size_t a, size_t b) { return eigenvalues[a] > eigenvalues[b]; });
    
    // Construir base proyectada
    int M_eff = std::min(config.M_trunc, static_cast<int>(eigenvalues.size()));
    for (int m = 0; m < M_eff; ++m) {
        int idx = indices[m];
        double lambda_sqrt = std::sqrt(eigenvalues[idx]);
        
        for (int i = 0; i < dim; ++i) {
            Complex sum = 0.0;
            for (int j = 0; j < config.S; ++j) {
                sum += Data[i][j] * eigenvectors[j][idx];
            }
            P[i][m] = sum / lambda_sqrt;
        }
    }
    
    std::cout << "Proyección construida: dim=" << dim << " -> M=" << M_eff << std::endl;
}

std::vector<MatrixXc> QuantumIsingSolver::timeEvolution(const VectorXc& psi0, 
                                                      const std::vector<double>& t_list) const {
    std::cout << "Calculando evolución temporal..." << std::endl;
    
    // Hamiltoniano efectivo H_eff = P^† H P
    MatrixXc P_dag = hermitianConjugate(P);
    MatrixXc H_P = matrixMultiply(H_dense, P);
    MatrixXc H_eff = matrixMultiply(P_dag, H_P);
    
    // Estado inicial proyectado
    VectorXc psi0_proj(config.M_trunc, 0.0);
    for (int m = 0; m < config.M_trunc; ++m) {
        for (int i = 0; i < dim; ++i) {
            psi0_proj[m] += std::conj(P[i][m]) * psi0[i];
        }
    }
    
    // Normalizar estado proyectado
    double norm = MathUtils::norm(psi0_proj);
    for (int m = 0; m < config.M_trunc; ++m) {
        psi0_proj[m] /= norm;
    }
    
    // Evolución temporal
    std::vector<MatrixXc> states(t_list.size(), MatrixXc(1, VectorXc(dim, 0.0)));
    
    for (size_t t_idx = 0; t_idx < t_list.size(); ++t_idx) {
        double t = t_list[t_idx];
        
        // Operador de evolución U_eff = exp(-i H_eff t)
        MatrixXc U_eff = matrixExponential(H_eff, -1.0j * t);
        
        // Evolucionar estado en subespacio
        VectorXc psi_t_proj(config.M_trunc, 0.0);
        for (int i = 0; i < config.M_trunc; ++i) {
            for (int j = 0; j < config.M_trunc; ++j) {
                psi_t_proj[i] += U_eff[i][j] * psi0_proj[j];
            }
        }
        
        // Levantar al espacio completo
        for (int i = 0; i < dim; ++i) {
            Complex sum = 0.0;
            for (int m = 0; m < config.M_trunc; ++m) {
                sum += P[i][m] * psi_t_proj[m];
            }
            states[t_idx][0][i] = sum;
        }
    }
    
    return states;
}

MatrixXc QuantumIsingSolver::buildSzTotal() const {
    MatrixXc Sz_total(dim, VectorXc(dim, 0.0));
    
    for (int i = 0; i < config.N; ++i) {
        MatrixXc term(dim, VectorXc(dim, 1.0));
        
        for (int k = 0; k < config.N; ++k) {
            MatrixXc op = (k == i) ? PauliMatrices::sz : PauliMatrices::id2;
            term = MathUtils::kron(term, op);
        }
        
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                Sz_total[row][col] += term[row][col];
            }
        }
    }
    
    return Sz_total;
}

double QuantumIsingSolver::computeMagnetization(const VectorXc& state) const {
    MatrixXc Sz_total = buildSzTotal();
    Complex expectation = 0.0;
    
    for (int i = 0; i < dim; ++i) {
        Complex sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            sum += Sz_total[i][j] * state[j];
        }
        expectation += std::conj(state[i]) * sum;
    }
    
    return std::real(expectation) / config.N;
}

// Implementación de utilidades matemáticas
namespace MathUtils {
    MatrixXc kron(const MatrixXc& A, const MatrixXc& B) {
        size_t a_rows = A.size(), a_cols = A[0].size();
        size_t b_rows = B.size(), b_cols = B[0].size();
        
        MatrixXc result(a_rows * b_rows, VectorXc(a_cols * b_cols, 0.0));
        
        for (size_t i = 0; i < a_rows; ++i) {
            for (size_t j = 0; j < a_cols; ++j) {
                for (size_t k = 0; k < b_rows; ++k) {
                    for (size_t l = 0; l < b_cols; ++l) {
                        result[i * b_rows + k][j * b_cols + l] = A[i][j] * B[k][l];
                    }
                }
            }
        }
        
        return result;
    }
    
    void eigenDecomposition(const MatrixXc& A, std::vector<double>& eigenvalues, 
                          MatrixXc& eigenvectors) {
        // Implementación simplificada - en la práctica usarías LAPACK/Eigen
        size_t n = A.size();
        eigenvalues.resize(n);
        eigenvectors = MatrixXc(n, VectorXc(n, 0.0));
        
        // Para demostración - valores propios diagonales
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = std::real(A[i][i]);  // Aproximación
            eigenvectors[i][i] = 1.0;
        }
    }
    
    double norm(const VectorXc& v) {
        double sum = 0.0;
        for (const auto& elem : v) {
            sum += std::norm(elem);
        }
        return std::sqrt(sum);
    }
    
    Complex dotProduct(const VectorXc& a, const VectorXc& b) {
        Complex result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += std::conj(a[i]) * b[i];
        }
        return result;
    }
}

// Implementación de métodos de QuantumIsingSolver
MatrixXc QuantumIsingSolver::matrixExponential(const MatrixXc& A, double t) {
    // Aproximación de Taylor para exp(A*t)
    int n = A.size();
    MatrixXc result(n, VectorXc(n, 0.0));
    MatrixXc term(n, VectorXc(n, 0.0));
    
    // Inicializar con identidad
    for (int i = 0; i < n; ++i) {
        result[i][i] = 1.0;
        term[i][i] = 1.0;
    }
    
    // Serie de Taylor (primeros 10 términos)
    const int max_terms = 10;
    for (int k = 1; k <= max_terms; ++k) {
        term = matrixMultiply(term, A);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                term[i][j] *= (t / k);
                result[i][j] += term[i][j];
            }
        }
    }
    
    return result;
}

MatrixXc QuantumIsingSolver::matrixMultiply(const MatrixXc& A, const MatrixXc& B) {
    int n = A.size(), m = B[0].size(), p = B.size();
    MatrixXc result(n, VectorXc(m, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

MatrixXc QuantumIsingSolver::hermitianConjugate(const MatrixXc& A) {
    int n = A.size(), m = A[0].size();
    MatrixXc result(m, VectorXc(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[j][i] = std::conj(A[i][j]);
        }
    }
    
    return result;
}

// Implementación de Visualizer
namespace Visualizer {
    void saveMatrix(const MatrixXc& mat, const std::string& filename) {
        std::ofstream file(filename);
        for (const auto& row : mat) {
            for (size_t j = 0; j < row.size(); ++j) {
                file << std::real(row[j]);
                if (j < row.size() - 1) file << ",";
            }
            file << "\n";
        }
    }
    
    void saveVector(const std::vector<double>& vec, const std::string& filename) {
        std::ofstream file(filename);
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) file << "\n";
        }
    }
    
    void saveComplexVector(const VectorXc& vec, const std::string& filename) {
        std::ofstream file(filename);
        for (size_t i = 0; i < vec.size(); ++i) {
            file << std::real(vec[i]) << "," << std::imag(vec[i]) << "\n";
        }
    }
}