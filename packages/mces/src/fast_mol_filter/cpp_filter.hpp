#ifndef CPP_FILTER_HPP
#define CPP_FILTER_HPP

#include <vector>
#include <string>
#include <map>
#include <GraphMol/ROMol.h>
// A map from an atom type (int) to a list of bond weights (doubles).
// The list of weights is pre-sorted in descending order.
using AtomWeightsMap = std::map<int, std::vector<double>>;

// Data for a single atom (node) needed for the cost calculation.
struct AtomData {
    AtomWeightsMap atom_weights;
    double total_weight;
};

// Holds all pre-computed information for a single molecule.
// This is the efficient representation you requested.
struct PrecomputedMol {
    // Maps an atom type (int) to a list of atom indices of that type.
    std::map<int, std::vector<unsigned int>> atom_types_to_indices;
    // A vector of data for each atom, indexed by the atom's original index.
    std::vector<AtomData> atom_data_vec;
};

// This will be the single entry point from Cython.
std::vector<double> calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list);

// New function for the `filter2` use case.
double calculate_total_cost_symmetric(const std::vector<std::string>& smiles_list);

std::vector<double> filter2_batch_symmetric(const std::vector<PrecomputedMol>& mols);

// Helper function to create a PrecomputedMol from a SMILES string.
// This will be called from Cython.
PrecomputedMol precompute_mol_data(const std::string& smiles);

#endif // CPP_FILTER_HPP