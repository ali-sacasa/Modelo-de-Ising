#include "hpp_Solver_D.E.s.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace Eigen;

int main() {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    
    SystemConfig config;
    config.N = 8;
    config.J = 1.0;
    config.h = 0.8;
    config.S = 400; 
    config.M_trunc = 30;
    config.T = 4.0;
    config.nt = 81;
    config.generate_time_list();
    config.BETA_MAX = 3.0;
    config.n_beta = 10;
    
    config.set_num_threads(0);  // Si no se especifica el número de hilos, se usa el máximo disponible
    
    config.print_config();
    
    std::cout << "\n=== INFORMACIÓN DE PARALELIZACIÓN ===\n";
    std::cout << "Threads OpenMP disponibles: " << omp_get_max_threads() << "\n";
    std::cout << "Threads configurados: " << config.num_threads << "\n";

    
    std::cout << "\n=== Construyendo Hamiltoniano ===\n";
    auto H_components = QuantumOperators::build_ising_hamiltonian_components(
        config.N, config.J, config.h, config.periodic);
    SparseMatrixXcd H_ZZ = H_components.first;
    SparseMatrixXcd H_X = H_components.second;
    SparseMatrixXcd H_total = H_ZZ + H_X;
    
    int dim = 1 << config.N;
    std::cout << "Dimensión del espacio: " << dim << std::endl;
    // Estado inicial |psi0>
    VectorXcd psi0 = VectorXcd::Zero(dim);
    psi0(0) = 1.0;

    std::cout << "\nMuestreando estados coherentes del sistema\n";
    auto coherent_start = high_resolution_clock::now();
    
    std::vector<Complex> z_list = CoherentStates::sample_cp1_fibonacci(config.S);
    MatrixXcd Data = CoherentStates::build_product_coherent_vectors(config.N, z_list);
    
    auto coherent_end = high_resolution_clock::now();
    auto coherent_duration = duration_cast<milliseconds>(coherent_end - coherent_start);
    std::cout << "Tiempo de construcción: " << coherent_duration.count() / 1000.0 << " s\n";
    
    // Optimización con Flujo de Ricci
    std::cout << "\nOptimizando con Flujo de Ricci\n";
    auto ricci_start = high_resolution_clock::now();
    
    RicciFlow ricci_flow(config);
    std::vector<double> beta_list(config.n_beta);
    double db = config.BETA_MAX / (config.n_beta - 1);
    for (int i = 0; i < config.n_beta; ++i) {
        beta_list[i] = 0.01 + i * db; // Comenzar en 0.01, puede cambiarse
    }
    
    auto ricci_results = ricci_flow.optimize_projection(H_total, psi0, Data, beta_list);
    
    auto ricci_end = high_resolution_clock::now();
    auto ricci_duration = duration_cast<milliseconds>(ricci_end - ricci_start);
    
    std::cout << "\nMejor beta encontrado: " << ricci_results.best_beta << std::endl;
    std::cout << "Fidelidad óptima: " << ricci_results.best_fidelity << std::endl;
    std::cout << "Dimensión efectiva: " << ricci_results.best_P.cols() 
              << "/" << dim << " (" 
              << 100.0 * ricci_results.best_P.cols() / dim << "%)" << std::endl;
    std::cout << "Tiempo de optimización: " << ricci_duration.count() / 1000.0 << " s\n";
    
    // Guardar resultados del flujo de Ricci
    CSVExporter::save_vector_csv(ricci_results.beta_list, "ricci_beta_list.csv", "beta");
    CSVExporter::save_vector_csv(ricci_results.fidelities, "ricci_fidelities.csv", "fidelity");
    

    std::cout << "\nEvolución temporal con polinomios Chebyshev\n";
    auto evolution_start = high_resolution_clock::now();
    
    auto evolution_results = ChebyshevPropagator::project_and_evolve(
        H_total, ricci_results.best_P, psi0, config.t_list, config.cheb_order);
    
    MatrixXcd states_proj = evolution_results.first;
    MatrixXcd states_full = evolution_results.second;
    
    auto evolution_end = high_resolution_clock::now();
    auto evolution_duration = duration_cast<milliseconds>(evolution_end - evolution_start);
    std::cout << "Tiempo de evolución: " << evolution_duration.count() / 1000.0 << " s\n";
    
    std::cout << "\nCalculando energías\n";
    auto obs_start = high_resolution_clock::now();
    
    // Energía total
    std::vector<double> energies_full = EnergyAnalyzer::compute_energy_evolution(states_full, H_total);
    std::vector<double> energies_proj = EnergyAnalyzer::compute_energy_evolution(states_proj, H_total);
    
    // Componentes de energía
    auto energy_components_full = EnergyAnalyzer::compute_energy_components_evolution(
        states_full, H_ZZ, H_X);
    auto energy_components_proj = EnergyAnalyzer::compute_energy_components_evolution(
        states_proj, H_ZZ, H_X);
    
    std::vector<double> E_ZZ_full = energy_components_full.first;
    std::vector<double> E_X_full = energy_components_full.second;
    std::vector<double> E_ZZ_proj = energy_components_proj.first;
    std::vector<double> E_X_proj = energy_components_proj.second;
    
    // Cálculo de la magnetización
    std::cout << "\nCalculando magnetización\n";
    SparseMatrixXcd Sz_total = QuantumOperators::build_Sz_total(config.N);
    MatrixXcd Sz_dense = MatrixXcd(Sz_total);
    
    std::vector<double> magnetization_full(config.nt);
    std::vector<double> magnetization_proj(config.nt);
    
    #pragma omp parallel for
    for (int i = 0; i < config.nt; ++i) {
        VectorXcd state_full = states_full.col(i);
        VectorXcd state_proj = states_proj.col(i);
        
        Complex mag_full = state_full.adjoint() * (Sz_dense * state_full);
        Complex mag_proj = state_proj.adjoint() * (Sz_dense * state_proj);
        
        magnetization_full[i] = mag_full.real() / config.N;
        magnetization_proj[i] = mag_proj.real() / config.N;
    }
    
    // Densidad de energía local
    auto h_local_list = QuantumOperators::build_local_energy_operators(
        config.N, config.J, config.h, config.periodic);
    
    MatrixXcd local_energy_full = EnergyAnalyzer::compute_local_energy_density(
        states_full, h_local_list);
    MatrixXcd local_energy_proj = EnergyAnalyzer::compute_local_energy_density(
        states_proj, h_local_list);
    
    // Varianza de energía
    std::cout << "\nCalculando varianza de energía\n";
    std::vector<double> variance_full = EnergyAnalyzer::compute_energy_variance(states_full, H_total);
    std::vector<double> variance_proj = EnergyAnalyzer::compute_energy_variance(states_proj, H_total);
    
    auto obs_end = high_resolution_clock::now();
    auto obs_duration = duration_cast<milliseconds>(obs_end - obs_start);
    std::cout << "Tiempo de energías: " << obs_duration.count() / 1000.0 << " s\n";
    
    
    std::cout << "\nGuardando resultados en archivos CSV\n";
    
    // Configuración
    std::ofstream config_file("config.csv");
    config_file << "parameter,value\n";
    config_file << "N," << config.N << "\n";
    config_file << "J," << config.J << "\n";
    config_file << "h," << config.h << "\n";
    config_file << "S," << config.S << "\n";
    config_file << "M_trunc," << config.M_trunc << "\n";
    config_file << "T," << config.T << "\n";
    config_file << "nt," << config.nt << "\n";
    config_file << "num_threads," << config.num_threads << "\n";  // *** AÑADIDO ***
    config_file << "best_beta," << ricci_results.best_beta << "\n";
    config_file << "best_fidelity," << ricci_results.best_fidelity << "\n";
    config_file.close();
    
    // Tiempos
    std::ofstream time_file("time.csv");
    time_file << "time\n";
    for (double t : config.t_list) {
    	time_file << t << "\n";
    }
    time_file.close();
    
    // Energías
    CSVExporter::save_energy_results_csv(
        config.t_list, energies_full, energies_proj,
        E_ZZ_full, E_X_full, E_ZZ_proj, E_X_proj,
        "energy_results.csv");
    
    // Magnetización
    CSVExporter::save_magnetization_csv(
        config.t_list, magnetization_full, magnetization_proj,
        "magnetization.csv");
    
    // Varianza
    std::ofstream variance_file("variance.csv");
    variance_file << "time,variance_full,variance_proj\n";
    for (int i = 0; i < config.nt; ++i) {
        variance_file << config.t_list[i] << ","
                      << variance_full[i] << ","
                      << variance_proj[i] << "\n";
    }
    variance_file.close();
    

    // Densidad de energía local
    CSVExporter::save_complex_matrix_csv(local_energy_full, "local_energy_full.csv");
    CSVExporter::save_complex_matrix_csv(local_energy_proj, "local_energy_proj.csv");
    
    // Estados (primeros 5 estados base para visualización)
    int max_basis = std::min(5, dim);
    MatrixXcd states_full_subset = states_full.topRows(max_basis).transpose();
    MatrixXcd states_proj_subset = states_proj.topRows(max_basis).transpose();
    
    CSVExporter::save_complex_matrix_csv(states_full_subset, "states_full_amplitudes.csv");
    CSVExporter::save_complex_matrix_csv(states_proj_subset, "states_proj_amplitudes.csv");
    
    // Rendimiento y error
    std::cout << "\nMétricas de error\n";

    // Error en energía y en magnetización
    double max_energy_error = 0.0;
    double mean_energy_error = 0.0;
    double max_mag_error = 0.0;
    
    for (int i = 0; i < config.nt; ++i) {
        double error = std::abs(energies_full[i] - energies_proj[i]);
        max_energy_error = std::max(max_energy_error, error);
        mean_energy_error += error;
        
        double mag_error = std::abs(magnetization_full[i] - magnetization_proj[i]);
        max_mag_error = std::max(max_mag_error, mag_error);
    }
    mean_energy_error /= config.nt;

    
    // Fidelidad durante la evolución
    std::vector<double> fidelity_evolution(config.nt);
    #pragma omp parallel for
    for (int i = 0; i < config.nt; ++i) {
        VectorXcd state_full = states_full.col(i);
        VectorXcd state_proj = states_proj.col(i);
        Complex overlap = state_proj.dot(state_full);
        fidelity_evolution[i] = std::norm(overlap);
    }

    // Fidelidades
    std::ofstream fidelity_file("fidelity_evolution.csv");
    fidelity_file << "fidelity\n";
    for (double fid : fidelity_evolution) {
        fidelity_file << fid << "\n";
    }
    fidelity_file.close();


    // Guardar métricas
    std::ofstream metrics_file("metrics.csv");
    metrics_file << "metric,value\n";
    metrics_file << "max_energy_error," << max_energy_error << "\n";
    metrics_file << "mean_energy_error," << mean_energy_error << "\n";
    metrics_file << "max_magnetization_error," << max_mag_error << "\n";
    metrics_file << "initial_fidelity," << ricci_results.best_fidelity << "\n";
    metrics_file << "average_fidelity," 
                 << std::accumulate(fidelity_evolution.begin(), fidelity_evolution.end(), 0.0) / config.nt 
                 << "\n";
    metrics_file.close();
    
    // Resumen y tiempo de ejecución

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    std::cout << "\n=== PROCESO COMPLETADO ===\n";
    std::cout << "Tiempo total: " << duration.count() / 1000.0 << " s\n";
    std::cout << "\nDesglose de tiempos:\n";
    std::cout << "  - Estados coherentes: " << coherent_duration.count() / 1000.0 << " s\n";
    std::cout << "  - Optimización Ricci: " << ricci_duration.count() / 1000.0 << " s\n";
    std::cout << "  - Evolución temporal: " << evolution_duration.count() / 1000.0 << " s\n";
    std::cout << "  - Cálculo observables: " << obs_duration.count() / 1000.0 << " s\n";
    std::cout << "\nThreads usados: " << config.num_threads << "\n";
    std::cout << "\nArchivos CSV generados:\n";
    std::cout << "  - config.csv: Configuración del sistema\n";
    std::cout << "  - time.csv: Lista de tiempos\n";
    std::cout << "  - energy_results.csv: Resultados de energía\n";
    std::cout << "  - magnetization.csv: Magnetización\n";
    std::cout << "  - variance.csv: Varianza de energía\n";
    std::cout << "  - local_energy_full.csv: Densidad de energía local (completo)\n";
    std::cout << "  - local_energy_proj.csv: Densidad de energía local (proyectado)\n";
    std::cout << "  - states_full_amplitudes.csv: Amplitudes de estados (completo)\n";
    std::cout << "  - states_proj_amplitudes.csv: Amplitudes de estados (proyectado)\n";
    std::cout << "  - fidelity_evolution.csv: Fidelidad durante la evolución\n";
    std::cout << "  - metrics.csv: Métricas de error\n";
    std::cout << "  - ricci_beta_list.csv: Valores de beta probados\n";
    std::cout << "  - ricci_fidelities.csv: Fidelidades correspondientes\n";
    std::cout << "Procesado finalizado, gracias.\n";
    
    return 0;
}
