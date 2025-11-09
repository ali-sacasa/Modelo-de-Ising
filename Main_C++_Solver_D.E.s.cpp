#include "hpp_Solver_D.E.s.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "=== SOLVER CUÁNTICO EN C++ - MODELO ISING ===" << std::endl;
    
    // Configuración del sistema
    SystemConfig config;
    config.N = 6;
    config.J = 1.0;
    config.h = 0.5;
    config.S = 100;
    config.M_trunc = 20;
    config.T = 5.0;
    config.nt = 50;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Crear solver
        QuantumIsingSolver solver(config);
        
        // Construir Hamiltoniano
        solver.buildIsingHamiltonian();
        
        // Muestrear estados coherentes
        MatrixXc Data = solver.buildCoherentVectors();
        
        // Construir proyección
        solver.buildProjection(Data);
        
        // Estado inicial |00...0⟩
        int dim = solver.getDimension();
        VectorXc psi0(dim, 0.0);
        psi0[0] = 1.0;
        
        // Tiempos de evolución
        std::vector<double> t_list(config.nt);
        for (int i = 0; i < config.nt; ++i) {
            t_list[i] = i * config.T / (config.nt - 1);
        }
        
        // Evolución temporal
        auto states = solver.timeEvolution(psi0, t_list);
        
        // Calcular magnetización
        std::vector<double> magnetization(t_list.size());
        for (size_t i = 0; i < t_list.size(); ++i) {
            magnetization[i] = solver.computeMagnetization(states[i][0]);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Guardar resultados
        Visualizer::saveVector(t_list, "tiempos.csv");
        Visualizer::saveVector(magnetization, "magnetizacion.csv");
        
        // Resultados
        std::cout << "\n=== RESULTADOS ===" << std::endl;
        std::cout << "Tiempo de ejecución: " << duration.count() << " ms" << std::endl;
        std::cout << "Dimensión del sistema: " << dim << std::endl;
        std::cout << "Puntos temporales: " << t_list.size() << std::endl;
        
        std::cout << "\nMagnetización en diferentes tiempos:" << std::endl;
        std::cout << "t=0: " << magnetization[0] << std::endl;
        std::cout << "t=" << t_list[t_list.size()/2] << ": " << magnetization[t_list.size()/2] << std::endl;
        std::cout << "t=" << t_list.back() << ": " << magnetization.back() << std::endl;
        
        std::cout << "\nDatos guardados en:" << std::endl;
        std::cout << "- tiempos.csv" << std::endl;
        std::cout << "- magnetizacion.csv" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n Simulación completada" << std::endl;
    return 0;
}

/* Posible forma para correr: g++ -std=c++17 -O3 -I. Main_C++_Solver_D.E.s.cpp Cpp_Solver_D.E.s.cpp -o quantum_solver
./quantum_solver*/