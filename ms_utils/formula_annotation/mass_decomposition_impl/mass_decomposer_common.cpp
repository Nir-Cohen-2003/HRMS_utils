#include "mass_decomposer_common.hpp"
#include <stdexcept>

MassDecomposer::MassDecomposer(const Formula& min_bounds, const Formula& max_bounds)
    : min_bounds_(min_bounds), max_bounds_(max_bounds), 
      precision_(1.0 / 5963.337687), is_initialized_(false) {
    // No longer need to initialize atomic masses or element indices here
}

bool MassDecomposer::check_dbe(const Formula& formula, double min_dbe, double max_dbe) const {
    int c_count = formula[FormulaAnnotation::C];
    int h_count = formula[FormulaAnnotation::H];
    int n_count = formula[FormulaAnnotation::N];
    int p_count = formula[FormulaAnnotation::P];
    int x_count = formula[FormulaAnnotation::F] + formula[FormulaAnnotation::Cl] + 
                  formula[FormulaAnnotation::Br] + formula[FormulaAnnotation::I];

    if (c_count == 0) return false;

    double dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0;
    
    if (dbe < min_dbe || dbe > max_dbe) return false;
    if (std::abs(dbe - std::round(dbe)) > 1e-8) return false;
    
    return true;
}

bool MassDecomposer::check_hetero_ratio(const Formula& formula, double max_ratio) const {
    int c_count = formula[FormulaAnnotation::C];
    if (c_count == 0) return false;
    
    int hetero_count = 0;
    for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
        if (i != FormulaAnnotation::C && i != FormulaAnnotation::H) {
            hetero_count += formula[i];
        }
    }
    
    return static_cast<double>(hetero_count) / static_cast<double>(c_count) <= max_ratio;
}

bool MassDecomposer::check_chemical_constraints(const Formula& formula, 
                                              const DecompositionParams& params) const {
    // Get element counts
    // int c_count = formula[FormulaAnnotation::C];
    
    // // Skip if no carbon
    // if (c_count == 0) return false;
    
    // Check DBE constraint
    if (!check_dbe(formula, params.min_dbe, params.max_dbe)) {
        return false;
    }

    // Check heteroatom ratio constraint
    if (params.max_hetero_ratio < 100.0) {
        if (!check_hetero_ratio(formula, params.max_hetero_ratio)) {
            return false;
        }
    }
    
    return true;
}

std::vector<Formula> MassDecomposer::decompose(double target_mass, const DecompositionParams& params) {
    if (!is_initialized_) {
        init_money_changing();
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
            double exact_mass = 0.0;
            for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
                exact_mass += result[i] * FormulaAnnotation::ATOMIC_MASSES[i];
            }

            // Check floating-point mass error
            double mass_error = std::abs(exact_mass - target_mass);
            if (mass_error > tolerance) {
                continue;
            }
            // Use unified constraint checking
            if (!check_chemical_constraints(result, params)) {
                continue;
            }
            results.push_back(result);
            if (static_cast<int>(results.size()) >= params.max_results) break;
        }
        if (static_cast<int>(results.size()) >= params.max_results) break;
    }
    return results;
}