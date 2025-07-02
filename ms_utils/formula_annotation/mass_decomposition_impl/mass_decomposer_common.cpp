#include "mass_decomposer_common.hpp"
#include <stdexcept>


MassDecomposer::MassDecomposer(const std::vector<Element>& elements, const std::string& strategy)
    : elements_(elements), precision_(1.0 / 5963.337687), is_initialized_(false),
      c_idx_(-1), h_idx_(-1), n_idx_(-1), p_idx_(-1), f_idx_(-1), 
      cl_idx_(-1), br_idx_(-1), i_idx_(-1) {
    
    init_atomic_masses();
    
    if (strategy == "money_changing") {
        init_money_changing();
    } else {
        init_recursive();
    }
}

void MassDecomposer::init_atomic_masses() {
    atomic_masses_ = {
        {"C", 12.0000000}, {"H", 1.0078250}, {"O", 15.9949146}, {"N", 14.0030740},
        {"P", 30.9737620}, {"S", 31.9720718}, {"F", 18.9984032}, {"Cl", 34.9688527},
        {"Br", 78.9183376}, {"I", 126.9044719}, {"Si", 27.9769271}, {"Na", 22.9897693},
        {"K", 38.9637069}, {"Ca", 39.9625912}, {"Mg", 23.9850423}, {"Fe", 55.9349421},
        {"Zn", 63.9291466}, {"Se", 79.9165218}, {"B", 11.0093054}, {"Al", 26.9815386}
    };
}

bool MassDecomposer::check_dbe(const Formula& formula, double min_dbe, double max_dbe) const {
    int c_count = formula.count("C") ? formula.at("C") : 0;
    int h_count = formula.count("H") ? formula.at("H") : 0;
    int n_count = formula.count("N") ? formula.at("N") : 0;
    int p_count = formula.count("P") ? formula.at("P") : 0;
    int x_count = (formula.count("F") ? formula.at("F") : 0) +
                  (formula.count("Cl") ? formula.at("Cl") : 0) +
                  (formula.count("Br") ? formula.at("Br") : 0) +
                  (formula.count("I") ? formula.at("I") : 0);

    if (c_count == 0) return false;

    double dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0;
    
    if (dbe < min_dbe || dbe > max_dbe) return false;
    if (std::abs(dbe - std::round(dbe)) > 1e-8) return false;
    
    return true;
}

bool MassDecomposer::check_hetero_ratio(const Formula& formula, double max_ratio) const {
    int c_count = formula.count("C") ? formula.at("C") : 0;
    if (c_count == 0) return false;
    
    int hetero_count = 0;
    for (const auto& pair : formula) {
        if (pair.first != "C" && pair.first != "H") {
            hetero_count += pair.second;
        }
    }
    
    return static_cast<double>(hetero_count) / static_cast<double>(c_count) <= max_ratio;
}

bool MassDecomposer::check_chemical_constraints(const std::vector<int>& formula, 
                                              const DecompositionParams& params) const {
    // Get element counts
    int c_count = (c_idx_ >= 0) ? formula[c_idx_] : 0;
    int h_count = (h_idx_ >= 0) ? formula[h_idx_] : 0;
    int n_count = (n_idx_ >= 0) ? formula[n_idx_] : 0;
    int p_count = (p_idx_ >= 0) ? formula[p_idx_] : 0;
    
    // Skip if no carbon
    if (c_count == 0) return false;
    
    // Calculate halogen count
    int x_count = 0;
    if (f_idx_ >= 0) x_count += formula[f_idx_];
    if (cl_idx_ >= 0) x_count += formula[cl_idx_];
    if (br_idx_ >= 0) x_count += formula[br_idx_];
    if (i_idx_ >= 0) x_count += formula[i_idx_];
    
    // Check DBE constraint
    double dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0;
    if (dbe < params.min_dbe || dbe > params.max_dbe || 
        std::abs(dbe - std::round(dbe)) > 1e-8) {
        return false;
    }

    // Check heteroatom ratio constraint
    if (params.max_hetero_ratio < 100.0) {
        int hetero_count = 0;
        for (size_t i = 0; i < formula.size(); ++i) {
            if (static_cast<int>(i) != c_idx_ && static_cast<int>(i) != h_idx_) {
                hetero_count += formula[i];
            }
        }
        
        if (static_cast<double>(hetero_count) / static_cast<double>(c_count) > params.max_hetero_ratio) {
            return false;
        }
    }
    
    return true;
}

std::vector<Formula> MassDecomposer::decompose(double target_mass, const DecompositionParams& params) {
    if (params.strategy == "money_changing") {
        if (!is_initialized_) {
            discretize_masses();
            divide_by_gcd();
            calc_ert();
            compute_errors();
            is_initialized_ = true;
        }
        
        double tolerance = target_mass * params.tolerance_ppm / 1e6;
        std::pair<long long, long long> bounds = integer_bound(target_mass - tolerance, target_mass + tolerance);
        long long start = bounds.first;
        long long end = bounds.second;
        
        std::vector<Formula> results;
        for (long long mass = start; mass <= end; ++mass) {
            auto mass_results = integer_decompose(mass);
            for (const auto& result : mass_results) {
                // Convert result to vector<int> for constraint checking
                std::vector<int> formula_vec(weights_.size(), 0);
                double exact_mass = 0.0;
                for (size_t i = 0; i < weights_.size(); ++i) {
                    auto it = result.find(weights_[i].symbol);
                    if (it != result.end()) {
                        formula_vec[i] = it->second;
                        exact_mass += it->second * atomic_masses_[weights_[i].symbol];
                    }
                }
                // Check floating-point mass error
                double mass_error = std::abs(exact_mass - target_mass);
                if (mass_error > tolerance) {
                    continue;
                }
                // Use unified constraint checking
                if (!check_chemical_constraints(formula_vec, params)) {
                    continue;
                }
                results.push_back(result);
                if (static_cast<int>(results.size()) >= params.max_results) break;
            }
            if (static_cast<int>(results.size()) >= params.max_results) break;
        }
        return results;
    } else {
        // Recursive strategy
        initialize_residue_tables();
        std::vector<int> formula(elements_.size(), 0);
        std::vector<Formula> results;
        double tolerance = target_mass * params.tolerance_ppm / 1e6;
        decompose_recursive_impl(formula, 0.0, 0, target_mass, tolerance, params, results);
        return results;
    }
}

std::vector<Element> MassDecomposer::create_bounds_from_formula(const Formula& formula) const {
    std::vector<Element> fragment_bounds;
    
    // Create bounds based on the precursor formula
    for (const auto& element : elements_) {
        Element frag_element = element;
        auto it = formula.find(element.symbol);
        if (it != formula.end()) {
            frag_element.min_count = 0;
            frag_element.max_count = it->second;  // Max is what's in the precursor
        } else {
            frag_element.min_count = 0;
            frag_element.max_count = 0;  // Element not in precursor, so not in fragments
        }
        fragment_bounds.push_back(frag_element);
    }
    
    return fragment_bounds;
}

std::vector<std::pair<std::string, int>> MassDecomposer::formula_to_pairs(const Formula& formula) const {
    std::vector<std::pair<std::string, int>> pairs;
    for (const auto& pair : formula) {
        pairs.push_back({pair.first, pair.second});
    }
    return pairs;
}