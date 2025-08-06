#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "cpp_filter.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <algorithm>
#include <stdexcept>
#include <set>

// Include the Hungarian algorithm implementation
#include <munkres.h>

// Real implementation of the LAP solver using munkres-cpp
double solve_lap(const std::vector<std::vector<double>>& cost_matrix) {
    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return 0.0;
    }

    // The munkres-cpp library uses its own Matrix class.
    // We need to convert our std::vector<std::vector<double>>.
    size_t rows = cost_matrix.size();
    size_t cols = cost_matrix[0].size();
    Matrix<double> matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = cost_matrix[i][j];
        }
    }

    // Create a Munkres object and solve the assignment problem.
    Munkres<double> m;
    m.solve(matrix);

    double total_cost = 0.0;
    // The 'solve' method modifies the matrix in-place.
    // The assigned pairs are marked with a 0.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (matrix(i, j) == 0) {
                // BUG: This assumes 0 means assignment, but what if original cost was 0?
                total_cost += cost_matrix[i][j];
            }
        }
    }
    return total_cost;
}


PrecomputedMol precompute_mol_data(const std::string& smiles) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol) {
        throw std::runtime_error("Failed to parse SMILES: " + smiles);
    }

    PrecomputedMol p_mol;
    p_mol.atom_data_vec.resize(mol->getNumAtoms());

    for (const auto atom : mol->atoms()) {
        unsigned int atom_idx = atom->getIdx();
        int atom_type = atom->getAtomicNum();
        p_mol.atom_types_to_indices[atom_type].push_back(atom_idx);

        AtomData& atom_data = p_mol.atom_data_vec[atom_idx];
        atom_data.total_weight = 0.0;

        for (const auto& neighbor : mol->atomNeighbors(atom)) {
            const RDKit::Bond* bond = mol->getBondBetweenAtoms(atom_idx, neighbor->getIdx());
            double weight = bond->getBondTypeAsDouble();
            int neighbor_type = neighbor->getAtomicNum();
            atom_data.atom_weights[neighbor_type].push_back(weight);
            atom_data.total_weight += weight;
        }
        
        atom_data.total_weight /= 2.0;

        for (auto& pair : atom_data.atom_weights) {
            std::sort(pair.second.rbegin(), pair.second.rend());
        }
    }
    return p_mol;
}

double node_cost(unsigned int node1_idx, const PrecomputedMol& mol1, unsigned int node2_idx, const PrecomputedMol& mol2) {
    const auto& weights1_map = mol1.atom_data_vec[node1_idx].atom_weights;
    const auto& weights2_map = mol2.atom_data_vec[node2_idx].atom_weights;

    std::set<int> all_atom_types;
    for (const auto& pair : weights1_map) all_atom_types.insert(pair.first);
    for (const auto& pair : weights2_map) all_atom_types.insert(pair.first);

    double cost = 0.0;
    for (int atom_type : all_atom_types) {
        auto it1 = weights1_map.find(atom_type);
        auto it2 = weights2_map.find(atom_type);

        const std::vector<double>& w1 = (it1 != weights1_map.end()) ? it1->second : std::vector<double>();
        const std::vector<double>& w2 = (it2 != weights2_map.end()) ? it2->second : std::vector<double>();

        size_t min_len = std::min(w1.size(), w2.size());
        for (size_t i = 0; i < min_len; ++i) {
            cost += std::abs(w1[i] - w2[i]);
        }
        
        for (size_t i = min_len; i < w1.size(); ++i) cost += w1[i];
        for (size_t i = min_len; i < w2.size(); ++i) cost += w2[i];
    }
    return cost / 2.0;
}

double calculate_pair_distance(const PrecomputedMol& mol1, const PrecomputedMol& mol2) {
    double total_cost = 0.0;
    
    std::set<int> all_types;
    for (const auto& pair : mol1.atom_types_to_indices) all_types.insert(pair.first);
    for (const auto& pair : mol2.atom_types_to_indices) all_types.insert(pair.first);

    for (int atom_type : all_types) {
        auto it1 = mol1.atom_types_to_indices.find(atom_type);
        auto it2 = mol2.atom_types_to_indices.find(atom_type);

        const std::vector<unsigned int>& nodes1 = (it1 != mol1.atom_types_to_indices.end()) ? it1->second : std::vector<unsigned int>();
        const std::vector<unsigned int>& nodes2 = (it2 != mol2.atom_types_to_indices.end()) ? it2->second : std::vector<unsigned int>();

        if (nodes1.empty()) {
            for (unsigned int n2_idx : nodes2) total_cost += mol2.atom_data_vec[n2_idx].total_weight;
            continue;
        }
        if (nodes2.empty()) {
            for (unsigned int n1_idx : nodes1) total_cost += mol1.atom_data_vec[n1_idx].total_weight;
            continue;
        }

        size_t n1 = nodes1.size();
        size_t n2 = nodes2.size();
        size_t max_size = std::max(n1, n2);
        std::vector<std::vector<double>> cost_matrix(max_size, std::vector<double>(max_size, 0.0));

        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                cost_matrix[i][j] = node_cost(nodes1[i], mol1, nodes2[j], mol2);
            }
        }

        if (n1 < n2) {
            for (size_t i = n1; i < max_size; ++i) {
                for (size_t j = 0; j < n2; ++j) {
                    cost_matrix[i][j] = mol2.atom_data_vec[nodes2[j]].total_weight;
                }
            }
        } else if (n2 < n1) {
            for (size_t i = 0; i < n1; ++i) {
                for (size_t j = n2; j < max_size; ++j) {
                    cost_matrix[i][j] = mol1.atom_data_vec[nodes1[i]].total_weight;
                }
            }
        }
        
        total_cost += solve_lap(cost_matrix);
    }
    return total_cost;
}

double calculate_total_cost_symmetric(const std::vector<std::string>& smiles_list) {
    size_t n = smiles_list.size();
    if (n < 2) {
        return 0.0;
    }
    std::vector<PrecomputedMol> precomputed_mols(n);

    // Stage 1: Pre-computation (in parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        precomputed_mols[i] = precompute_mol_data(smiles_list[i]);
    }

    // Stage 2: Pairwise distance calculation and summation (in parallel)
    double total_cost = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:total_cost)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            total_cost += calculate_pair_distance(precomputed_mols[i], precomputed_mols[j]);
        }
    }
    return total_cost;
}

std::vector<double> calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list) {
    size_t n = smiles_list.size();
    std::vector<PrecomputedMol> precomputed_mols(n);

    // Stage 1: Pre-computation (in parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        // RDKit is not thread-safe for certain operations, but creating molecules
        // from SMILES is generally safe. If issues arise, this may need a critical section.
        precomputed_mols[i] = precompute_mol_data(smiles_list[i]);
    }

    // Stage 2: Pairwise distance calculation (also parallelized internally)
    return filter2_batch_symmetric(precomputed_mols);
}

std::vector<double> filter2_batch_symmetric(const std::vector<PrecomputedMol>& mols) {
    size_t n = mols.size();
    std::vector<double> results(n * n, 0.0); // Initialize with 0

        int error_count = 0;

        #pragma omp parallel for schedule(dynamic) reduction(+:error_count)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                if (i == j) {
                    results[i * n + j] = 0.0;
                    continue;
                }
                double dist = calculate_pair_distance(mols[i], mols[j]);
                // Add bounds checking
                if (std::isfinite(dist) && dist >= 0 && dist <= 10000.0) {
                    results[i * n + j] = dist;
                    results[j * n + i] = dist;
                } else {
                    // Handle invalid results or errors
                    results[i * n + j] = 0.0;
                    results[j * n + i] = 0.0;
                    if (dist > 10000.0) {
                        error_count++;
                    }
                }
            }
        }
        if (error_count > 0) {
            std::cerr << "Warning: " << error_count << " pairwise distances exceeded 10000." << std::endl;
        }

    return results;
}