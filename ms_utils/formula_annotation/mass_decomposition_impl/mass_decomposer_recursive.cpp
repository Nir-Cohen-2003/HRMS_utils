#include "mass_decomposer_common.hpp"

void MassDecomposer::init_recursive() {
    // Sort elements by mass (heaviest first for better pruning)
    std::sort(elements_.begin(), elements_.end(),
              [this](const Element& a, const Element& b) {
                  return atomic_masses_[a.symbol] > atomic_masses_[b.symbol];
              });
    
    for (size_t i = 0; i < elements_.size(); ++i) {
        if (elements_[i].symbol == "C") c_idx_ = i;
        else if (elements_[i].symbol == "H") h_idx_ = i;
        else if (elements_[i].symbol == "N") n_idx_ = i;
        else if (elements_[i].symbol == "P") p_idx_ = i;
        else if (elements_[i].symbol == "F") f_idx_ = i;
        else if (elements_[i].symbol == "Cl") cl_idx_ = i;
        else if (elements_[i].symbol == "Br") br_idx_ = i;
        else if (elements_[i].symbol == "I") i_idx_ = i;
    }
}

void MassDecomposer::initialize_residue_tables() {
    min_residues_.resize(elements_.size());
    max_residues_.resize(elements_.size());
    
    double min_mass = 0.0, max_mass = 0.0;
    for (int i = static_cast<int>(elements_.size()) - 1; i >= 0; --i) {
        min_mass += elements_[i].min_count * atomic_masses_.at(elements_[i].symbol);
        max_mass += elements_[i].max_count * atomic_masses_.at(elements_[i].symbol);
        min_residues_[i] = min_mass;
        max_residues_[i] = max_mass;
    }
}

bool MassDecomposer::can_reach_target(double current_mass, int level, double target_mass, double tolerance) const {
    if (level >= static_cast<int>(elements_.size())) {
        return std::abs(current_mass - target_mass) <= tolerance;
    }
    
    double remaining_min = current_mass + min_residues_[level];
    double remaining_max = current_mass + max_residues_[level];
    
    return (target_mass - tolerance <= remaining_max && 
            remaining_min <= target_mass + tolerance);
}

void MassDecomposer::decompose_recursive_impl(std::vector<int>& formula, double current_mass, 
                                            int level, double target_mass, double tolerance,
                                            const DecompositionParams& params,
                                            std::vector<Formula>& results) const {
    // Base case: all elements processed
    if (level >= static_cast<int>(elements_.size())) {
        double mass_diff = std::abs(current_mass - target_mass);
        if (mass_diff <= tolerance) {
            if (check_chemical_constraints(formula, params)) {
                Formula result;
                for (size_t i = 0; i < formula.size(); ++i) {
                    if (formula[i] > 0) {
                        result[elements_[i].symbol] = formula[i];
                    }
                }
                results.push_back(result);
            }
        }
        return;
    }
    
    // Early termination
    if (static_cast<int>(results.size()) >= params.max_results) return;
    if (!can_reach_target(current_mass, level, target_mass, tolerance)) return;
    
    // Get element properties
    double element_mass = atomic_masses_.at(elements_[level].symbol);
    int min_count = elements_[level].min_count;
    int max_count = elements_[level].max_count;
    
    // Additional pruning
    if (element_mass > 0) {
        int max_possible_count = static_cast<int>((target_mass + tolerance - current_mass) / element_mass);
        max_count = std::min(max_count, max_possible_count);
    }
    
    if (max_count < min_count) return;
    
    // Try all possible counts
    for (int count = min_count; count <= max_count; ++count) {
        double new_mass = current_mass + count * element_mass;
        if (new_mass > target_mass + tolerance) break;
        
        formula[level] = count;
        decompose_recursive_impl(formula, new_mass, level + 1, target_mass, tolerance, params, results);
    }
}