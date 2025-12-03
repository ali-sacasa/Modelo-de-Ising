#include "hpp_Solver_D.E.s.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/FFT>

using namespace Eigen;

// Configuración del sistema
SystemConfig::SystemConfig() :
    N(8), J(1.0), h(0.8), periodic(false),
    S(500), M_trunc(50),
    T(8.0), nt(161), cheb_order(120),
    BETA_MAX(5.0), n_beta(15),
    output_dir("results/")
{
    generate_time_list();
}

void SystemConfig::generate_time_list() {
    t_list.resize(nt);
    double dt = T / (nt - 1);
    for (int i = 0; i < nt; ++i) {
        t_list[i] = i * dt;
    }
}

void SystemConfig::print_config() const {
    std::cout << "=== Configuración del Sistema ===\n";
    std::cout << "N: " << N << " espines\n";
    std::cout << "J: " << J << ", h: " << h << "\n";
    std::cout << "S: " << S << " estados coherentes\n";
    std::cout << "M_trunc: " << M_trunc << "\n";
    std::cout << "T: " << T << ", nt: " << nt << "\n";
    std::cout << "BETA_MAX: " << BETA_MAX << ", n_beta: " << n_beta << "\n";
    std::cout << "Output directory: " << output_dir << "\n";
}

// Implementación de QuantumOperators
MatrixXcd QuantumOperators::sigma_x() {
    MatrixXcd sx(2, 2);
    sx << 0.0, 1.0,
          1.0, 0.0;
    return sx;
}

MatrixXcd QuantumOperators::sigma_y() {
    MatrixXcd sy(2, 2);
    sy << 0.0, -I,
          I, 0.0;
    return sy;
}

MatrixXcd QuantumOperators::sigma_z() {
    MatrixXcd sz(2, 2);
    sz << 1.0, 0.0,
          0.0, -1.0;
    return sz;
}

MatrixXcd QuantumOperators::identity_2() {
    return MatrixXcd::Identity(2, 2);
}

MatrixXcd QuantumOperators::kron_list(const std::vector<MatrixXcd>& mats) {
    if (mats.empty()) {
        return MatrixXcd::Identity(1, 1);
    }
    
    MatrixXcd result = mats[0];
    for (size_t i = 1; i < mats.size(); ++i) {
        // Producto de Kronecker
        int rows_result = result.rows();
        int cols_result = result.cols();
        int rows_mat = mats[i].rows();
        int cols_mat = mats[i].cols();
        
        MatrixXcd kron_result(rows_result * rows_mat, cols_result * cols_mat);
        
        for (int r1 = 0; r1 < rows_result; ++r1) {
            for (int c1 = 0; c1 < cols_result; ++c1) {
                kron_result.block(r1 * rows_mat, c1 * cols_mat, rows_mat, cols_mat) = 
                    result(r1, c1) * mats[i];
            }
        }
        result = kron_result;
    }
    return result;
}

SparseMatrixXcd QuantumOperators::kron_list_sparse(const std::vector<SparseMatrixXcd>& mats) {
    if (mats.empty()) {
        SparseMatrixXcd I(1, 1);
        I.insert(0, 0) = 1.0;
        return I;
    }
    
    SparseMatrixXcd result = mats[0];
    for (size_t i = 1; i < mats.size(); ++i) {
        // Producto de Kronecker para matrices sparse
        result = Eigen::kroneckerProduct(result, mats[i]).eval();
    }
    return result;
}

SparseMatrixXcd QuantumOperators::build_ising_hamiltonian(int N, double J, double h, bool periodic) {
    int dim = 1 << N;  // 2^N
    SparseMatrixXcd H(dim, dim);
    
    // Para construcción eficiente
    std::vector<Triplet<Complex>> triplets;
    triplets.reserve(dim * (N + (periodic ? N : N-1)));
    
    // Parte de interacción ZZ
    int n_bonds = periodic ? N : N - 1;
    for (int i = 0; i < n_bonds; ++i) {
        int j = (i + 1) % N;
        
        // Construir operador sigma_z^i sigma_z^j
        std::vector<SparseMatrixXcd> ops(N);
        for (int k = 0; k < N; ++k) {
            if (k == i || k == j) {
                ops[k] = sigma_z().sparseView();
            } else {
                ops[k] = identity_2().sparseView();
            }
        }
        
        SparseMatrixXcd sz_sz = kron_list_sparse(ops);
        
        // Añadir términos a la lista de triplets
        for (int k = 0; k < sz_sz.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(sz_sz, k); it; ++it) {
                triplets.push_back(Triplet<Complex>(it.row(), it.col(), -J * it.value()));
            }
        }
    }
    
    // Parte del campo transversal
    for (int i = 0; i < N; ++i) {
        std::vector<SparseMatrixXcd> ops(N);
        for (int k = 0; k < N; ++k) {
            if (k == i) {
                ops[k] = sigma_x().sparseView();
            } else {
                ops[k] = identity_2().sparseView();
            }
        }
        
        SparseMatrixXcd sx_i = kron_list_sparse(ops);
        
        for (int k = 0; k < sx_i.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(sx_i, k); it; ++it) {
                triplets.push_back(Triplet<Complex>(it.row(), it.col(), -h * it.value()));
            }
        }
    }
    
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
}

std::pair<SparseMatrixXcd, SparseMatrixXcd> 
QuantumOperators::build_ising_hamiltonian_components(int N, double J, double h, bool periodic) {
    int dim = 1 << N;
    SparseMatrixXcd H_ZZ(dim, dim);
    SparseMatrixXcd H_X(dim, dim);
    
    std::vector<Triplet<Complex>> triplets_zz, triplets_x;
    
    // Parte de interacción ZZ
    int n_bonds = periodic ? N : N - 1;
    for (int i = 0; i < n_bonds; ++i) {
        int j = (i + 1) % N;
        
        std::vector<SparseMatrixXcd> ops(N);
        for (int k = 0; k < N; ++k) {
            if (k == i || k == j) {
                ops[k] = sigma_z().sparseView();
            } else {
                ops[k] = identity_2().sparseView();
            }
        }
        
        SparseMatrixXcd sz_sz = kron_list_sparse(ops);
        
        for (int k = 0; k < sz_sz.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(sz_sz, k); it; ++it) {
                triplets_zz.push_back(Triplet<Complex>(it.row(), it.col(), -J * it.value()));
            }
        }
    }
    
    // Parte del campo transversal
    for (int i = 0; i < N; ++i) {
        std::vector<SparseMatrixXcd> ops(N);
        for (int k = 0; k < N; ++k) {
            if (k == i) {
                ops[k] = sigma_x().sparseView();
            } else {
                ops[k] = identity_2().sparseView();
            }
        }
        
        SparseMatrixXcd sx_i = kron_list_sparse(ops);
        
        for (int k = 0; k < sx_i.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(sx_i, k); it; ++it) {
                triplets_x.push_back(Triplet<Complex>(it.row(), it.col(), -h * it.value()));
            }
        }
    }
    
    H_ZZ.setFromTriplets(triplets_zz.begin(), triplets_zz.end());
    H_X.setFromTriplets(triplets_x.begin(), triplets_x.end());
    
    return std::make_pair(H_ZZ, H_X);
}

SparseMatrixXcd QuantumOperators::build_Sz_total(int N) {
    int dim = 1 << N;
    SparseMatrixXcd Sz_total(dim, dim);
    
    std::vector<Triplet<Complex>> triplets;
    
    for (int i = 0; i < N; ++i) {
        std::vector<SparseMatrixXcd> ops(N);
        for (int k = 0; k < N; ++k) {
            if (k == i) {
                ops[k] = sigma_z().sparseView();
            } else {
                ops[k] = identity_2().sparseView();
            }
        }
        
        SparseMatrixXcd sz_i = kron_list_sparse(ops);
        
        for (int k = 0; k < sz_i.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(sz_i, k); it; ++it) {
                triplets.push_back(Triplet<Complex>(it.row(), it.col(), it.value()));
            }
        }
    }
    
    Sz_total.setFromTriplets(triplets.begin(), triplets.end());
    return Sz_total;
}

std::vector<SparseMatrixXcd> QuantumOperators::build_local_energy_operators(
    int N, double J, double h, bool periodic) {
    
    std::vector<SparseMatrixXcd> h_local(N);
    int n_bonds = periodic ? N : N - 1;
    
    for (int i = 0; i < N; ++i) {
        SparseMatrixXcd h_site = SparseMatrixXcd(1 << N, 1 << N);
        std::vector<Triplet<Complex>> triplets;
        
        // Contribución del campo en el sitio i
        {
            std::vector<SparseMatrixXcd> ops(N);
            for (int k = 0; k < N; ++k) {
                if (k == i) {
                    ops[k] = sigma_x().sparseView();
                } else {
                    ops[k] = identity_2().sparseView();
                }
            }
            SparseMatrixXcd sx_i = kron_list_sparse(ops);
            
            for (int k = 0; k < sx_i.outerSize(); ++k) {
                for (SparseMatrixXcd::InnerIterator it(sx_i, k); it; ++it) {
                    triplets.push_back(Triplet<Complex>(it.row(), it.col(), -h * it.value()));
                }
            }
        }
        
        // Contribuciones de enlaces (mitad para cada sitio)
        // Enlace a la derecha
        if (i < N - 1 || periodic) {
            int j = (i + 1) % N;
            std::vector<SparseMatrixXcd> ops(N);
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) {
                    ops[k] = sigma_z().sparseView();
                } else {
                    ops[k] = identity_2().sparseView();
                }
            }
            SparseMatrixXcd sz_sz = kron_list_sparse(ops);
            
            for (int k = 0; k < sz_sz.outerSize(); ++k) {
                for (SparseMatrixXcd::InnerIterator it(sz_sz, k); it; ++it) {
                    triplets.push_back(Triplet<Complex>(it.row(), it.col(), -J * it.value() / 2.0));
                }
            }
        }
        
        // Enlace a la izquierda
        if (i > 0 || periodic) {
            int j = (i - 1 + N) % N;
            std::vector<SparseMatrixXcd> ops(N);
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) {
                    ops[k] = sigma_z().sparseView();
                } else {
                    ops[k] = identity_2().sparseView();
                }
            }
            SparseMatrixXcd sz_sz = kron_list_sparse(ops);
            
            for (int k = 0; k < sz_sz.outerSize(); ++k) {
                for (SparseMatrixXcd::InnerIterator it(sz_sz, k); it; ++it) {
                    triplets.push_back(Triplet<Complex>(it.row(), it.col(), -J * it.value() / 2.0));
                }
            }
        }
        
        h_site.setFromTriplets(triplets.begin(), triplets.end());
        h_local[i] = h_site;
    }
    
    return h_local;
}

// Implementación de CoherentStates
VectorXcd CoherentStates::coherent_state(const Complex& z) {
    VectorXcd state(2);
    double norm = std::sqrt(1.0 + std::norm(z));
    state << 1.0, z;
    return state / norm;
}

std::vector<Complex> CoherentStates::sample_cp1_fibonacci(int S) {
    std::vector<Complex> z_list(S);
    double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    
    for (int i = 0; i < S; ++i) {
        double index = static_cast<double>(i);
        double phi = 2.0 * PI * index * (1.0 / golden_ratio);
        double theta = std::acos(1.0 - 2.0 * (index + 0.5) / S);
        z_list[i] = std::tan(theta / 2.0) * std::exp(I * phi);
    }
    
    return z_list;
}

MatrixXcd CoherentStates::build_product_coherent_vectors(int N, const std::vector<Complex>& z_list) {
    int S = z_list.size();
    int dim = 1 << N;  // 2^N
    MatrixXcd Data(dim, S);
    
    for (int j = 0; j < S; ++j) {
        VectorXcd single = coherent_state(z_list[j]);
        VectorXcd psi = single;
        
        // Producto tensorial para N espines
        for (int i = 1; i < N; ++i) {
            int rows_psi = psi.rows();
            int rows_single = single.rows();
            VectorXcd kron_vec(rows_psi * rows_single);
            
            for (int r1 = 0; r1 < rows_psi; ++r1) {
                kron_vec.segment(r1 * rows_single, rows_single) = psi(r1) * single;
            }
            psi = kron_vec;
        }
        Data.col(j) = psi;
    }
    
    // Normalizar cada columna
    for (int j = 0; j < S; ++j) {
        double norm = Data.col(j).norm();
        if (norm > 1e-12) {
            Data.col(j) /= norm;
        }
    }
    
    return Data;
}

MatrixXcd CoherentStates::compute_gram_matrix(const MatrixXcd& Data) {
    return Data.adjoint() * Data;
}

MatrixXcd CoherentStates::build_projection_from_gram(
    const MatrixXcd& Data, int M, std::vector<double>& eigenvalues) {
    
    MatrixXcd G = compute_gram_matrix(Data);
    
    // Diagonalizar la matriz de Gram
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(G);
    if (eigensolver.info() != Success) {
        std::cerr << "Error al diagonalizar la matriz de Gram!" << std::endl;
        return MatrixXcd::Zero(Data.rows(), 1);
    }
    
    // Obtener autovalores y autovectores
    VectorXd evals = eigensolver.eigenvalues().real();
    MatrixXcd evecs = eigensolver.eigenvectors();
    
    // Ordenar en orden descendente
    std::vector<int> indices(evals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&evals](int i, int j) { return evals(i) > evals(j); });
    
    int M_eff = std::min(M, (int)evals.size());
    eigenvalues.resize(M_eff);
    
    // Construir matriz de proyección
    MatrixXcd P(Data.rows(), M_eff);
    
    for (int m = 0; m < M_eff; ++m) {
        int idx = indices[m];
        eigenvalues[m] = std::max(evals(idx), 1e-12);
        VectorXcd v = evecs.col(idx);
        P.col(m) = (Data * v) / std::sqrt(eigenvalues[m]);
    }
    
    // Ortogonalización de Gram-Schmidt
    for (int m = 0; m < M_eff; ++m) {
        VectorXcd col = P.col(m);
        for (int n = 0; n < m; ++n) {
            col -= P.col(n) * P.col(n).dot(col);
        }
        col.normalize();
        P.col(m) = col;
    }
    
    return P;
}

// Implementación del flujo de Ricci
RicciFlow::RicciFlow(const SystemConfig& cfg) : config(cfg), rng(42) {}

std::vector<double> RicciFlow::get_thermal_weights(
    const SparseMatrixXcd& H, const MatrixXcd& Data, double beta) {
    
    int S = Data.cols();
    std::vector<double> weights(S, 0.0);
    
    if (beta < 1e-8) {
        std::fill(weights.begin(), weights.end(), 1.0 / S);
        return weights;
    }
    
    // Convertir a matriz densa para exponencial (para sistemas pequeños)
    MatrixXcd H_dense = MatrixXcd(H);
    
    // Calcular operador de densidad térmica: exp(-beta H)
    // Usar descomposición espectral para exponencial
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(H_dense);
    if (eigensolver.info() != Success) {
        std::cerr << "Error al diagonalizar H para pesos térmicos!" << std::endl;
        return weights;
    }
    
    VectorXd evals = eigensolver.eigenvalues().real();
    MatrixXcd evecs = eigensolver.eigenvectors();
    
    // Construir exp(-beta H)
    MatrixXcd exp_neg_beta_H = MatrixXcd::Zero(H_dense.rows(), H_dense.cols());
    for (int i = 0; i < evals.size(); ++i) {
        exp_neg_beta_H += std::exp(-beta * evals(i)) * evecs.col(i) * evecs.col(i).adjoint();
    }
    
    // Calcular pesos
    double total_weight = 0.0;
    for (int k = 0; k < S; ++k) {
        VectorXcd Zk = Data.col(k);
        Complex weight = Zk.adjoint() * (exp_neg_beta_H * Zk);
        weights[k] = std::max(weight.real(), 1e-12);
        total_weight += weights[k];
    }
    
    // Normalizar
    if (total_weight > 1e-12) {
        for (double& w : weights) {
            w /= total_weight;
        }
    }
    
    return weights;
}

MatrixXcd RicciFlow::build_weighted_projection(
    const MatrixXcd& Data, int M, 
    const std::vector<double>& weights,
    std::vector<double>& eigenvalues) {
    
    int S = Data.cols();
    int dim = Data.rows();
    
    // Crear matriz Data ponderada
    MatrixXcd D_tilde = Data;
    for (int j = 0; j < S; ++j) {
        D_tilde.col(j) *= std::sqrt(weights[j]);
    }
    
    // Matriz de Gram con regularización
    MatrixXcd G_tilde = D_tilde.adjoint() * D_tilde;
    G_tilde += 1e-8 * MatrixXcd::Identity(S, S);
    
    // Diagonalizar
    SelfAdjointEigenSolver<MatrixXcd> eigensolver(G_tilde);
    if (eigensolver.info() != Success) {
        std::cerr << "Error al diagonalizar G_tilde!" << std::endl;
        return MatrixXcd::Zero(dim, 1);
    }
    
    VectorXd evals = eigensolver.eigenvalues().real();
    MatrixXcd evecs = eigensolver.eigenvectors();
    
    // Ordenar en orden descendente
    std::vector<int> indices(S);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&evals](int i, int j) { return evals(i) > evals(j); });
    
    // Selección adaptativa de dimensión
    double total_variance = evals.sum();
    double cumulative = 0.0;
    int M_eff = 0;
    double threshold_ratio = 0.01;
    
    for (int i = 0; i < S; ++i) {
        int idx = indices[i];
        cumulative += evals(idx);
        if (cumulative / total_variance > (1.0 - threshold_ratio)) {
            M_eff = i + 1;
            break;
        }
    }
    M_eff = std::min(M_eff, M);
    M_eff = std::min(M_eff, S);
    
    eigenvalues.resize(M_eff);
    
    // Construir matriz de proyección
    MatrixXcd P(dim, M_eff);
    
    for (int m = 0; m < M_eff; ++m) {
        int idx = indices[m];
        eigenvalues[m] = std::max(evals(idx), 1e-12);
        VectorXcd v = evecs.col(idx);
        P.col(m) = (D_tilde * v) / std::sqrt(eigenvalues[m]);
    }
    
    // Ortogonalización
    for (int m = 0; m < M_eff; ++m) {
        VectorXcd col = P.col(m);
        for (int n = 0; n < m; ++n) {
            col -= P.col(n) * P.col(n).dot(col);
        }
        col.normalize();
        P.col(m) = col;
    }
    
    return P;
}

double RicciFlow::calculate_fidelity(const MatrixXcd& P, const VectorXcd& psi0) {
    VectorXcd psi0_proj = P.adjoint() * psi0;
    VectorXcd psi0_reconstructed = P * psi0_proj;
    Complex overlap = psi0_reconstructed.dot(psi0);
    return std::norm(overlap);
}

RicciFlow::RicciResults RicciFlow::optimize_projection(
    const SparseMatrixXcd& H, const VectorXcd& psi0, 
    const MatrixXcd& Data, const std::vector<double>& beta_list) {
    
    RicciResults results;
    results.beta_list = beta_list;
    results.fidelities.resize(beta_list.size());
    results.weights_list.resize(beta_list.size());
    
    double best_fidelity = 0.0;
    int best_idx = 0;
    
    std::cout << "Optimizando proyección con flujo de Ricci..." << std::endl;
    
    for (size_t i = 0; i < beta_list.size(); ++i) {
        double beta = beta_list[i];
        
        // 1. Calcular pesos térmicos
        std::vector<double> weights = get_thermal_weights(H, Data, beta);
        results.weights_list[i] = weights;
        
        // 2. Construir proyección ponderada
        std::vector<double> eigenvalues;
        MatrixXcd P = build_weighted_projection(Data, config.M_trunc, weights, eigenvalues);
        
        // 3. Calcular fidelidad
        double fid = calculate_fidelity(P, psi0);
        results.fidelities[i] = fid;
        
        // Guardar mejor proyección
        if (fid > best_fidelity) {
            best_fidelity = fid;
            best_idx = i;
            results.best_P = P;
            results.best_beta = beta;
            results.best_fidelity = fid;
            results.best_idx = best_idx;
        }
        
        std::cout << "  beta = " << beta 
                  << ", Fidelidad = " << fid 
                  << ", M_eff = " << P.cols() << std::endl;
    }
    
    // Construir Hamiltoniano efectivo para la mejor proyección
    MatrixXcd H_dense = MatrixXcd(H);
    MatrixXcd HP = H_dense * results.best_P;
    results.best_H_eff = results.best_P.adjoint() * HP;
    
    return results;
}

void RicciFlow::save_results_csv(const RicciResults& results, const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    file << "beta,fidelity\n";
    for (size_t i = 0; i < results.beta_list.size(); ++i) {
        file << results.beta_list[i] << "," << results.fidelities[i] << "\n";
    }
    file.close();
    
    std::cout << "Resultados guardados en: " << filename << std::endl;
}

// Implementación de ChebyshevPropagator
ChebyshevPropagator::ChebyshevPropagator(const SparseMatrixXcd& H_matrix, int M_order) 
    : H(H_matrix), M(M_order) {
    prepare_rescaled_hamiltonian();
}

void ChebyshevPropagator::prepare_rescaled_hamiltonian() {
    // Estimar extremos espectrales usando Lanczos
    try {
        // Convertir a matriz densa para sistemas pequeños
        MatrixXcd H_dense = MatrixXcd(H);
        SelfAdjointEigenSolver<MatrixXcd> eigensolver(H_dense);
        if (eigensolver.info() == Success) {
            VectorXd evals = eigensolver.eigenvalues();
            emin = evals.minCoeff();
            emax = evals.maxCoeff();
        } else {
            // Fallback
            emin = -10.0;
            emax = 10.0;
        }
    } catch (...) {
        emin = -10.0;
        emax = 10.0;
    }
    
    if (emax <= emin) {
        emax = emin + 1e-10;
    }
    
    a = (emax - emin) / 2.0;
    b = (emax + emin) / 2.0;
    
    // Crear matriz reescalada: (H - bI)/a
    int n = H.rows();
    SparseMatrixXcd I(n, n);
    I.setIdentity();
    
    H_rescaled = (H - b * I) * (1.0 / a);
}

VectorXcd ChebyshevPropagator::evolve(const VectorXcd& psi0, double t) const {
    // Coeficientes de Chebyshev usando funciones de Bessel
    std::vector<Complex> coeff(M + 1);
    Complex z = a * t;
    Complex pref = std::exp(-I * b * t);
    
    coeff[0] = pref * std::cyl_bessel_j(0, z.real());  // Aproximación
    for (int n = 1; n <= M; ++n) {
        coeff[n] = 2.0 * pref * std::cyl_bessel_j(n, z.real());
    }
    
    // Recurrencia de Chebyshev
    VectorXcd T0 = psi0;
    VectorXcd T1 = H_rescaled * psi0;
    VectorXcd psi_t = coeff[0] * T0 + coeff[1] * T1;
    
    for (int n = 2; n <= M; ++n) {
        VectorXcd Tn = 2.0 * H_rescaled * T1 - T0;
        psi_t += coeff[n] * Tn;
        T0 = T1;
        T1 = Tn;
    }
    
    return psi_t;
}

std::pair<MatrixXcd, MatrixXcd> ChebyshevPropagator::project_and_evolve(
    const SparseMatrixXcd& H, const MatrixXcd& P, const VectorXcd& psi0,
    const std::vector<double>& t_list, int cheb_order) {
    
    int nt = t_list.size();
    int dim = H.rows();
    int M_eff = P.cols();
    
    // Hamiltoniano efectivo
    MatrixXcd H_dense = MatrixXcd(H);
    MatrixXcd HP = H_dense * P;
    MatrixXcd H_eff = P.adjoint() * HP;
    
    // Estado inicial proyectado
    VectorXcd psi0_proj = P.adjoint() * psi0;
    psi0_proj.normalize();
    
    // Propagadores
    ChebyshevPropagator prop_eff(SparseMatrixXcd(H_eff.sparseView()), cheb_order);
    ChebyshevPropagator prop_full(H, cheb_order);
    
    // Estados evolucionados
    MatrixXcd states_proj(dim, nt);
    MatrixXcd states_full(dim, nt);
    
    std::cout << "Evolucionando con Chebyshev..." << std::endl;
    
    for (int i = 0; i < nt; ++i) {
        double t = t_list[i];
        
        // Evolución en subespacio proyectado
        VectorXcd psi_proj_t = prop_eff.evolve(psi0_proj, t);
        VectorXcd psi_proj_lifted = P * psi_proj_t;
        states_proj.col(i) = psi_proj_lifted;
        
        // Evolución completa
        VectorXcd psi_full_t = prop_full.evolve(psi0, t);
        states_full.col(i) = psi_full_t;
        
        if (i % 20 == 0) {
            std::cout << "  t = " << t << "/" << t_list.back() << std::endl;
        }
    }
    
    return std::make_pair(states_proj, states_full);
}

// =============================================================================
// Implementación de EnergyAnalyzer
// =============================================================================
std::vector<double> EnergyAnalyzer::compute_energy_evolution(
    const MatrixXcd& states, const SparseMatrixXcd& H) {
    
    int nt = states.cols();
    std::vector<double> energies(nt);
    MatrixXcd H_dense = MatrixXcd(H);
    
    for (int i = 0; i < nt; ++i) {
        VectorXcd state = states.col(i);
        Complex energy = state.adjoint() * (H_dense * state);
        energies[i] = energy.real();
    }
    
    return energies;
}

std::pair<std::vector<double>, std::vector<double>> 
EnergyAnalyzer::compute_energy_components_evolution(
    const MatrixXcd& states, 
    const SparseMatrixXcd& H_ZZ, 
    const SparseMatrixXcd& H_X) {
    
    int nt = states.cols();
    std::vector<double> E_ZZ(nt), E_X(nt);
    
    MatrixXcd H_ZZ_dense = MatrixXcd(H_ZZ);
    MatrixXcd H_X_dense = MatrixXcd(H_X);
    
    for (int i = 0; i < nt; ++i) {
        VectorXcd state = states.col(i);
        
        Complex e_zz = state.adjoint() * (H_ZZ_dense * state);
        Complex e_x = state.adjoint() * (H_X_dense * state);
        
        E_ZZ[i] = e_zz.real();
        E_X[i] = e_x.real();
    }
    
    return std::make_pair(E_ZZ, E_X);
}

MatrixXd EnergyAnalyzer::compute_local_energy_density(
    const MatrixXcd& states, 
    const std::vector<SparseMatrixXcd>& h_local_list) {
    
    int nt = states.cols();
    int N = h_local_list.size();
    MatrixXd local_energy(nt, N);
    
    for (int i = 0; i < nt; ++i) {
        VectorXcd state = states.col(i);
        
        for (int j = 0; j < N; ++j) {
            MatrixXcd h_local_dense = MatrixXcd(h_local_list[j]);
            Complex e_local = state.adjoint() * (h_local_dense * state);
            local_energy(i, j) = e_local.real();
        }
    }
    
    return local_energy;
}

std::vector<double> EnergyAnalyzer::compute_energy_variance(
    const MatrixXcd& states, const SparseMatrixXcd& H) {
    
    int nt = states.cols();
    std::vector<double> variance(nt);
    
    MatrixXcd H_dense = MatrixXcd(H);
    MatrixXcd H_squared = H_dense * H_dense;
    
    for (int i = 0; i < nt; ++i) {
        VectorXcd state = states.col(i);
        
        Complex H_exp = state.adjoint() * (H_dense * state);
        Complex H2_exp = state.adjoint() * (H_squared * state);
        
        variance[i] = (H2_exp - H_exp * H_exp).real();
    }
    
    return variance;
}

std::pair<std::vector<double>, std::vector<double>> 
EnergyAnalyzer::compute_energy_conservation_error(
    const std::vector<double>& energies, double initial_energy) {
    
    int nt = energies.size();
    std::vector<double> abs_error(nt), rel_error(nt);
    
    for (int i = 0; i < nt; ++i) {
        abs_error[i] = std::abs(energies[i] - initial_energy);
        rel_error[i] = abs_error[i] / (std::abs(initial_energy) + 1e-12);
    }
    
    return std::make_pair(abs_error, rel_error);
}

// Implementación de CSVExporter

void CSVExporter::save_vector_csv(const std::vector<double>& data, 
                                 const std::string& filename,
                                 const std::string& header) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    if (!header.empty()) {
        file << header << "\n";
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i < data.size() - 1) file << ",";
    }
    file << "\n";
    file.close();
}

void CSVExporter::save_complex_vector_csv(const std::vector<Complex>& data,
                                         const std::string& filename,
                                         const std::string& header) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    if (!header.empty()) {
        file << header << "\n";
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i].real() << "," << data[i].imag();
        if (i < data.size() - 1) file << ",";
    }
    file << "\n";
    file.close();
}

void CSVExporter::save_matrix_csv(const MatrixXd& data,
                                 const std::string& filename,
                                 const std::string& header) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    if (!header.empty()) {
        file << header << "\n";
    }
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            file << data(i, j);
            if (j < data.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void CSVExporter::save_complex_matrix_csv(const MatrixXcd& data,
                                         const std::string& filename,
                                         const std::string& header) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    if (!header.empty()) {
        file << header << "\n";
    }
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            file << data(i, j).real() << "," << data(i, j).imag();
            if (j < data.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void CSVExporter::save_energy_results_csv(
    const std::vector<double>& t_list,
    const std::vector<double>& energies_full,
    const std::vector<double>& energies_proj,
    const std::vector<double>& E_ZZ_full,
    const std::vector<double>& E_X_full,
    const std::vector<double>& E_ZZ_proj,
    const std::vector<double>& E_X_proj,
    const std::string& filename) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    file << "time,energy_full,energy_proj,E_ZZ_full,E_X_full,E_ZZ_proj,E_X_proj\n";
    
    for (size_t i = 0; i < t_list.size(); ++i) {
        file << t_list[i] << ","
             << energies_full[i] << ","
             << energies_proj[i] << ","
             << E_ZZ_full[i] << ","
             << E_X_full[i] << ","
             << E_ZZ_proj[i] << ","
             << E_X_proj[i] << "\n";
    }
    
    file.close();
    std::cout << "Resultados de energía guardados en: " << filename << std::endl;
}

void CSVExporter::save_magnetization_csv(
    const std::vector<double>& t_list,
    const std::vector<double>& magnetization_full,
    const std::vector<double>& magnetization_proj,
    const std::string& filename) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir archivo: " << filename << std::endl;
        return;
    }
    
    file << "time,magnetization_full,magnetization_proj\n";
    
    for (size_t i = 0; i < t_list.size(); ++i) {
        file << t_list[i] << ","
             << magnetization_full[i] << ","
             << magnetization_proj[i] << "\n";
    }
    
    file.close();
    std::cout << "Magnetización guardada en: " << filename << std::endl;
}