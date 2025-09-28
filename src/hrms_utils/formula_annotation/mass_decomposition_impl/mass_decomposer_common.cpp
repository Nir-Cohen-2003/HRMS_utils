#include "mass_decomposer_common.hpp"
#include <stdexcept>

std::string MassDecomposer::formula_to_string(const Formula& formula) {
    static const std::array<int, FormulaAnnotation::NUM_ELEMENTS> order = {
        FormulaAnnotation::C,
        FormulaAnnotation::H,
        FormulaAnnotation::B,
        FormulaAnnotation::N,
        FormulaAnnotation::O,
        FormulaAnnotation::F,
        FormulaAnnotation::Na,
        FormulaAnnotation::Si,
        FormulaAnnotation::P,
        FormulaAnnotation::S,
        FormulaAnnotation::Cl,
        FormulaAnnotation::K,
        FormulaAnnotation::As,
        FormulaAnnotation::Br,
        FormulaAnnotation::I
    };

    std::string out;
    out.reserve(64);
    for (int idx : order) {
        const int count = formula[idx];
        if (count <= 0) {
            continue;
        }
        const char* symbol = FormulaAnnotation::element_symbol_at(idx);
        out.append(symbol);
        if (count > 1) {
            out.append(std::to_string(count));
        }
    }
    return out;
}

MassDecomposer::MassDecomposer(const Formula& min_bounds, const Formula& max_bounds)
    : min_bounds_(min_bounds), max_bounds_(max_bounds), 
      precision_(1.0 / 5963.337687), is_initialized_(false) {
    // No longer need to initialize atomic masses or element indices here
}

inline bool MassDecomposer::check_dbe(const Formula& formula, double min_dbe, double max_dbe) const {
    int c_count = formula[FormulaAnnotation::C]+ formula[FormulaAnnotation::Si];
    int h_count = formula[FormulaAnnotation::H];
    int n_count = formula[FormulaAnnotation::N]+ formula[FormulaAnnotation::B]+ formula[FormulaAnnotation::As];
    int p_count = formula[FormulaAnnotation::P];
    int x_count = formula[FormulaAnnotation::F] + formula[FormulaAnnotation::Cl] + 
                  formula[FormulaAnnotation::Br] + formula[FormulaAnnotation::I];

    double dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0;
    
    if (dbe < min_dbe || dbe > max_dbe) return false;
    if (std::abs(dbe - std::round(dbe)) > 1e-8) return false;
    
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
    
    // Use max(target_mass, 200.0) for tolerance calculation, since below it stops being in relative terms
    double tolerance = std::max(target_mass, 200.0) * params.tolerance_ppm / 1e6;
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
            if (!check_dbe(result, params.min_dbe, params.max_dbe)) {
                continue;
            }
            results.push_back(result);
        }
    }
    return results;
}

std::vector<FormulaWithString> MassDecomposer::decompose_verbose(double target_mass, const DecompositionParams& params) {
    const auto formulas = decompose(target_mass, params);
    std::vector<FormulaWithString> verbose_results;
    verbose_results.reserve(formulas.size());
    for (const auto& formula : formulas) {
        verbose_results.push_back(FormulaWithString{formula, formula_to_string(formula)});
    }
    return verbose_results;
}