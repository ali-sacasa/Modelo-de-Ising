#ifndef hpp_Solver_D.E.s_hpp
#define hpp_Solver_D.E.s_hpp

#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Definición de tipos para facilitar el uso
typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> MatrixXcd;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> VectorXcd;
typedef Eigen::SparseMatrix<Complex, Eigen::RowMajor> SparseMatrixXcd;

// Constantes
const Complex I(0.0, 1.0);
const double PI = 3.14159265358979323846;

class SystemConfig {
public:
    // Parámetros físicos
    int N;                      // número de espines
    double J;                   // acoplamiento ZZ
    double h;                   // campo transversal (Sx)
    bool periodic;              // condiciones periódicas
    
    // Muestreo y proyección
    int S;                      // número de estados coherentes
    int M_trunc;                // dimensión del subespacio proyectado
    
    // Evolución temporal
    double T;                   // tiempo final
    int nt;                     // número de puntos temporales
    std::vector<double> t_list; // lista de tiempos
    int cheb_order;             // orden de Chebyshev
    
    // Flujo de Ricci
    double BETA_MAX;            // máximo parámetro de escala
    int n_beta;                 // número de puntos en beta
    
    // Configuración de salida
    std::string output_dir;     // directorio para archivos CSV
    
    // Constructor
    SystemConfig();
    
    // Métodos de utilidad
    void generate_time_list();
    void print_config() const;
};


class QuantumOperators {
public:
    // Matrices de Pauli
    static MatrixXcd sigma_x();
    static MatrixXcd sigma_y();
    static MatrixXcd sigma_z();
    static MatrixXcd identity_2();
    
    // Construcción de operadores
    static MatrixXcd kron_list(const std::vector<MatrixXcd>& mats);
    static SparseMatrixXcd kron_list_sparse(const std::vector<SparseMatrixXcd>& mats);
    
    // Hamiltoniano de Ising
    static SparseMatrixXcd build_ising_hamiltonian(int N, double J, double h, bool periodic);
    static std::pair<SparseMatrixXcd, SparseMatrixXcd> 
    build_ising_hamiltonian_components(int N, double J, double h, bool periodic);
    
    // Operadores de espín
    static SparseMatrixXcd build_Sz_total(int N);
    static SparseMatrixXcd build_Sx_total(int N);
    
    // Operadores locales de energía
    static std::vector<SparseMatrixXcd> build_local_energy_operators(
        int N, double J, double h, bool periodic);
};

class CoherentStates {
public:
    // Estado coherente para un solo espín
    static VectorXcd coherent_state(const Complex& z);
    
    // Muestreo de estados coherentes (Fibonacci lattice)
    static std::vector<Complex> sample_cp1_fibonacci(int S);
    
    // Construcción de estados producto
    static MatrixXcd build_product_coherent_vectors(
        int N, const std::vector<Complex>& z_list);
    
    // Calcular matriz de Gram
    static MatrixXcd compute_gram_matrix(const MatrixXcd& Data);
    
    // Construir proyección usando SVD/descomposición espectral
    static MatrixXcd build_projection_from_gram(
        const MatrixXcd& Data, int M, std::vector<double>& eigenvalues);
};

class RicciFlow {
private:
    SystemConfig config;
    std::mt19937 rng;
    
public:
    RicciFlow(const SystemConfig& cfg);
    
    // Pesos térmicos
    std::vector<double> get_thermal_weights(
        const SparseMatrixXcd& H, const MatrixXcd& Data, double beta);
    
    // Proyección ponderada
    MatrixXcd build_weighted_projection(
        const MatrixXcd& Data, int M, 
        const std::vector<double>& weights,
        std::vector<double>& eigenvalues);
    
    // Calcular fidelidad
    double calculate_fidelity(const MatrixXcd& P, const VectorXcd& psi0);
    
    // Optimizar proyección
    struct RicciResults {
        std::vector<double> beta_list;
        std::vector<double> fidelities;
        std::vector<std::vector<double>> weights_list;
        double best_beta;
        double best_fidelity;
        MatrixXcd best_P;
        MatrixXcd best_H_eff;
        int best_idx;
    };
    
    RicciResults optimize_projection(
        const SparseMatrixXcd& H, const VectorXcd& psi0, 
        const MatrixXcd& Data, const std::vector<double>& beta_list);
    
    // Guardar resultados
    void save_results_csv(const RicciResults& results, const std::string& filename) const;
};


class ChebyshevPropagator {
private:
    SparseMatrixXcd H;
    int M;
    double emin, emax, a, b;
    SparseMatrixXcd H_rescaled;
    
    void prepare_rescaled_hamiltonian();
    
public:
    ChebyshevPropagator(const SparseMatrixXcd& H_matrix, int M_order = 80);
    
    // Evolucionar estado
    VectorXcd evolve(const VectorXcd& psi0, double t) const;
    
    // Evolución completa
    static std::pair<MatrixXcd, MatrixXcd> project_and_evolve(
        const SparseMatrixXcd& H, const MatrixXcd& P, const VectorXcd& psi0,
        const std::vector<double>& t_list, int cheb_order);
};

class EnergyAnalyzer {
public:
    // Energía total
    static std::vector<double> compute_energy_evolution(
        const MatrixXcd& states, const SparseMatrixXcd& H);
    
    // Componentes de energía
    static std::pair<std::vector<double>, std::vector<double>> 
    compute_energy_components_evolution(
        const MatrixXcd& states, 
        const SparseMatrixXcd& H_ZZ, 
        const SparseMatrixXcd& H_X);
    
    // Densidad de energía local
    static MatrixXd compute_local_energy_density(
        const MatrixXcd& states, 
        const std::vector<SparseMatrixXcd>& h_local_list);
    
    // Varianza de energía
    static std::vector<double> compute_energy_variance(
        const MatrixXcd& states, const SparseMatrixXcd& H);
    
    // Error de conservación
    static std::pair<std::vector<double>, std::vector<double>> 
    compute_energy_conservation_error(
        const std::vector<double>& energies, double initial_energy);
};


class CSVExporter {
public:
    // Guardar vector 1D
    static void save_vector_csv(const std::vector<double>& data, 
                               const std::string& filename,
                               const std::string& header = "");
    
    // Guardar vector complejo 1D
    static void save_complex_vector_csv(const std::vector<Complex>& data,
                                       const std::string& filename,
                                       const std::string& header = "");
    
    // Guardar matriz 2D
    static void save_matrix_csv(const MatrixXd& data,
                               const std::string& filename,
                               const std::string& header = "");
    
    // Guardar matriz compleja 2D
    static void save_complex_matrix_csv(const MatrixXcd& data,
                                       const std::string& filename,
                                       const std::string& header = "");
    
    // Guardar resultados de energía
    static void save_energy_results_csv(
        const std::vector<double>& t_list,
        const std::vector<double>& energies_full,
        const std::vector<double>& energies_proj,
        const std::vector<double>& E_ZZ_full,
        const std::vector<double>& E_X_full,
        const std::vector<double>& E_ZZ_proj,
        const std::vector<double>& E_X_proj,
        const std::string& filename);
    
    // Guardar magnetización
    static void save_magnetization_csv(
        const std::vector<double>& t_list,
        const std::vector<double>& magnetization_full,
        const std::vector<double>& magnetization_proj,
        const std::string& filename);
};

#endif // hpp_Solver_D.E.s_hpp