#include "mass_decomposer_core.hpp"
#include <omp.h>
#include <climits>
#include <cstdlib>
#include <stdexcept>

MassDecomposer::MassDecomposer()
    : precision_(1.0 / 5963.337687), 
      c_idx_(-1), h_idx_(-1), n_idx_(-1), p_idx_(-1), f_idx_(-1), 
      cl_idx_(-1), br_idx_(-1), i_idx_(-1) {
    element_order_ = {"H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"};
    init_atomic_masses();
    element_masses_vec_.resize(element_order_.size());
    for(size_t i = 0; i < element_order_.size(); ++i) {
        element_masses_vec_[i] = atomic_masses_[element_order_[i]];
        if (element_order_[i] == "C") c_idx_ = i;
        else if (element_order_[i] == "H") h_idx_ = i;
        else if (element_order_[i] == "N") n_idx_ = i;
        else if (element_order_[i] == "P") p_idx_ = i;
        else if (element_order_[i] == "F") f_idx_ = i;
        else if (element_order_[i] == "Cl") cl_idx_ = i;
        else if (element_order_[i] == "Br") br_idx_ = i;
        else if (element_order_[i] == "I") i_idx_ = i;
    }
}

void MassDecomposer::init_atomic_masses() {
    atomic_masses_ = {
        {"H", 1.007825},
        {"B", 11.009305},
        {"C", 12.000000},
        {"N", 14.003074},
        {"O", 15.994915},
        {"F", 18.998403},
        {"Na", 22.989770},
        {"Si", 27.9769265},
        {"P", 30.973762},
        {"S", 31.972071},
        {"Cl", 34.96885271},
        {"K", 38.963707},
        {"As", 74.921596},
        {"Br", 78.918338},
        {"I", 126.904468}
    };
}

void MassDecomposer::init_elements(const BoundsArray& bounds) {
    weights_.resize(element_order_.size());
    for (size_t i = 0; i < element_order_.size(); ++i) {
        weights_[i].symbol = element_order_[i];
        weights_[i].mass = atomic_masses_[element_order_[i]];
        weights_[i].min_count = bounds[0][i];
        weights_[i].max_count = bounds[1][i];
    }
}

std::vector<FormulaArray> MassDecomposer::decompose(double target_mass, const BoundsArray& bounds, const DecompositionParams& params) {
    init_elements(bounds);
    if (params.strategy == "money_changing") {
        init_money_changing();
    } else {
        init_recursive();
    }

    std::vector<FormulaArray> results;
    double tolerance = target_mass * params.mass_accuracy_ppm / 1e6;

    if (params.strategy == "money_changing") {
        auto mass_bounds = integer_bound(target_mass - tolerance, target_mass + tolerance);
        for (long long m = mass_bounds.first; m <= mass_bounds.second; ++m) {
            auto integer_results = integer_decompose(m);
            for(auto& res : integer_results) {
                double exact_mass = 0;
                for(size_t i=0; i<res.size(); ++i) exact_mass += res[i] * element_masses_vec_[i];
                if (std::abs(exact_mass - target_mass) <= tolerance && check_chemical_constraints(res, params)){
                    results.push_back(res);
                }
            }
        }
    } else {
        std::vector<int> formula(element_order_.size(), 0);
        decompose_recursive_impl(formula, 0.0, 0, target_mass, tolerance, params, results);
    }
    return results;
}

// 1. Simple mass decomposition
std::vector<FormulaArray> MassDecomposer::decompose_mass(double target_mass, const BoundsArray& bounds, const DecompositionParams& params){
    return decompose(target_mass, bounds, params);
}

// 2. Spectrum decomposition with unknown precursor
std::vector<std::vector<FormulaArray>> MassDecomposer::decompose_spectrum(double precursor_mass, const std::vector<double>& fragment_masses, const BoundsArray& bounds, const DecompositionParams& params) {
    std::vector<FormulaArray> precursor_formulas = decompose(precursor_mass, bounds, params);
    std::vector<std::vector<FormulaArray>> results;
    for (const auto& precursor_formula : precursor_formulas) {
        BoundsArray fragment_bounds = {std::vector<int>(element_order_.size(), 0), precursor_formula};
        std::vector<FormulaArray> all_fragment_formulas;
        for (double fragment_mass : fragment_masses) {
            auto fragment_results = decompose(fragment_mass, fragment_bounds, params);
            all_fragment_formulas.insert(all_fragment_formulas.end(), fragment_results.begin(), fragment_results.end());
        }
        results.push_back(all_fragment_formulas);
    }
    return results;
}

// 3. Spectrum decomposition with known precursor
std::vector<std::vector<FormulaArray>> MassDecomposer::decompose_spectrum_known_precursor(const FormulaArray& precursor_formula, const std::vector<double>& fragment_masses, const DecompositionParams& params) {
    BoundsArray fragment_bounds = {std::vector<int>(element_order_.size(), 0), precursor_formula};
    std::vector<std::vector<FormulaArray>> results;
    for (double fragment_mass : fragment_masses) {
        results.push_back(decompose(fragment_mass, fragment_bounds, params));
    }
    return results;
}

// 4. Parallel mass decomposition
std::vector<std::vector<FormulaArray>> MassDecomposer::decompose_mass_parallel(const std::vector<double>& target_masses, const BoundsArray& uniform_bounds, const DecompositionParams& params) {
    std::vector<std::vector<FormulaArray>> all_results(target_masses.size());
    #pragma omp parallel for
    for (size_t i = 0; i < target_masses.size(); ++i) {
        MassDecomposer local_decomposer;
        all_results[i] = local_decomposer.decompose(target_masses[i], uniform_bounds, params);
    }
    return all_results;
}

std::vector<std::vector<FormulaArray>> MassDecomposer::decompose_mass_parallel(const std::vector<double>& target_masses, const std::vector<BoundsArray>& per_mass_bounds, const DecompositionParams& params) {
    std::vector<std::vector<FormulaArray>> all_results(target_masses.size());
    #pragma omp parallel for
    for (size_t i = 0; i < target_masses.size(); ++i) {
        MassDecomposer local_decomposer;
        all_results[i] = local_decomposer.decompose(target_masses[i], per_mass_bounds[i], params);
    }
    return all_results;
}

// 5. Parallel spectrum decomposition with unknown precursor
std::vector<std::vector<std::vector<FormulaArray>>> MassDecomposer::decompose_spectrum_parallel(const std::vector<Spectrum>& spectra, const BoundsArray& uniform_bounds, const DecompositionParams& params) {
    std::vector<std::vector<std::vector<FormulaArray>>> all_results(spectra.size());
    #pragma omp parallel for
    for (size_t i = 0; i < spectra.size(); ++i) {
        MassDecomposer local_decomposer;
        all_results[i] = local_decomposer.decompose_spectrum(spectra[i].precursor_mass, spectra[i].fragment_masses, uniform_bounds, params);
    }
    return all_results;
}

std::vector<std::vector<std::vector<FormulaArray>>> MassDecomposer::decompose_spectrum_parallel(const std::vector<SpectrumWithBounds>& spectra_with_bounds, const DecompositionParams& params) {
    std::vector<std::vector<std::vector<FormulaArray>>> all_results(spectra_with_bounds.size());
    #pragma omp parallel for
    for (size_t i = 0; i < spectra_with_bounds.size(); ++i) {
        MassDecomposer local_decomposer;
        all_results[i] = local_decomposer.decompose_spectrum(spectra_with_bounds[i].precursor_mass, spectra_with_bounds[i].fragment_masses, spectra_with_bounds[i].bounds, params);
    }
    return all_results;
}

// 6. Parallel spectrum decomposition with known precursor
std::vector<std::vector<std::vector<FormulaArray>>> MassDecomposer::decompose_spectrum_known_precursor_parallel(const std::vector<SpectrumWithKnownPrecursor>& spectra, const DecompositionParams& params) {
    std::vector<std::vector<std::vector<FormulaArray>>> all_results(spectra.size());
    #pragma omp parallel for
    for (size_t i = 0; i < spectra.size(); ++i) {
        MassDecomposer local_decomposer;
        all_results[i] = local_decomposer.decompose_spectrum_known_precursor(spectra[i].precursor_formula, spectra[i].fragment_masses, params);
    }
    return all_results;
}

void MassDecomposer::init_money_changing() {
    std::sort(weights_.begin(), weights_.end(), [](const Weight& a, const Weight& b) { return a.mass < b.mass; });
    discretize_masses();
    divide_by_gcd();
    calc_ert();
    compute_errors();
}

void MassDecomposer::init_recursive() {
    std::sort(weights_.begin(), weights_.end(), [](const Weight& a, const Weight& b) { return a.mass > b.mass; });
    initialize_residue_tables();
}

std::vector<FormulaArray> MassDecomposer::integer_decompose(long long mass) const {
    std::vector<FormulaArray> results;
    int k = static_cast<int>(weights_.size()) - 1;
    if (k < 0) return results;
    long long a = weights_[0].integer_mass;
    if (a <= 0) return results;
    std::vector<int> c(k + 1, 0);
    int i = k;
    long long m = mass;
    while (i <= k) {
        if (!decomposable(i, m, a)) {
            while (i <= k && !decomposable(i, m, a)) { m += c[i] * weights_[i].integer_mass; c[i] = 0; i++; }
            if (i <= k) { m -= weights_[i].integer_mass; c[i]++; }
        } else {
            while (i > 0 && decomposable(i - 1, m, a)) { i--; }
            if (i == 0) {
                if (a > 0) c[0] = static_cast<int>(m / a);
                else c[0] = 0;
                bool valid_formula = true;
                for (int j = 0; j <= k; ++j) {
                    if (c[j] < weights_[j].min_count || c[j] > weights_[j].max_count) { valid_formula = false; break; }
                }
                if (valid_formula) {
                    FormulaArray result(element_order_.size(), 0);
                    for(int j=0; j<=k; ++j) result[j] = c[j];
                    results.push_back(result);
                }
                i++;
            }
            while (i <= k && c[i] >= weights_[i].max_count) { m += c[i] * weights_[i].integer_mass; c[i] = 0; i++; }
            if (i <= k) { m -= weights_[i].integer_mass; c[i]++; }
        }
    }
    return results;
}

void MassDecomposer::decompose_recursive_impl(std::vector<int>& formula, double current_mass, int level, double target_mass, double tolerance, const DecompositionParams& params, std::vector<FormulaArray>& results) const {
    if (level >= static_cast<int>(weights_.size())) {
        if (std::abs(current_mass - target_mass) <= tolerance && check_chemical_constraints(formula, params)) {
            results.push_back(formula);
        }
        return;
    }
    if (static_cast<int>(results.size()) >= params.max_results) return;
    if (!can_reach_target(current_mass, level, target_mass, tolerance)) return;
    double element_mass = weights_[level].mass;
    int min_count = weights_[level].min_count;
    int max_count = weights_[level].max_count;
    for (int count = min_count; count <= max_count; ++count) {
        double new_mass = current_mass + count * element_mass;
        if (new_mass > target_mass + tolerance) break;
        if (level < static_cast<int>(formula.size())) { // Check to prevent out-of-bounds access
            formula[level] = count;
            decompose_recursive_impl(formula, new_mass, level + 1, target_mass, tolerance, params, results);
        }
    }
}

bool MassDecomposer::check_chemical_constraints(const FormulaArray& formula, const DecompositionParams& params) const {
    int c_count = (c_idx_ >= 0) ? formula[c_idx_] : 0;
    if (c_count == 0) return false;
    int h_count = (h_idx_ >= 0) ? formula[h_idx_] : 0;
    int n_count = (n_idx_ >= 0) ? formula[n_idx_] : 0;
    int p_count = (p_idx_ >= 0) ? formula[p_idx_] : 0;
    int x_count = 0;
    if (f_idx_ >= 0) x_count += formula[f_idx_];
    if (cl_idx_ >= 0) x_count += formula[cl_idx_];
    if (br_idx_ >= 0) x_count += formula[br_idx_];
    if (i_idx_ >= 0) x_count += formula[i_idx_];
    double dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0;
    if (dbe < params.min_dbe || dbe > params.max_dbe) return false;
    // Check if DBE is an integer or half-integer, allowing for small tolerance
    if (std::abs(dbe * 2.0 - std::round(dbe * 2.0)) > 1e-8) return false;

    if (params.max_hetero_ratio < 100.0) {
        int hetero_count = 0;
        for (size_t i = 0; i < formula.size(); ++i) {
            if (i != static_cast<size_t>(c_idx_) && i != static_cast<size_t>(h_idx_)) {
                hetero_count += formula[i];
            }
        }
        if (c_count > 0 && static_cast<double>(hetero_count) / c_count > params.max_hetero_ratio) return false;
    }
    return true;
}

void MassDecomposer::discretize_masses() {
    if (weights_.empty()) return;
    long long p = static_cast<long long>(1.0 / precision_);
    for (auto& w : weights_) {
        w.integer_mass = static_cast<long long>(round(w.mass * p));
    }
}

void MassDecomposer::divide_by_gcd() {
    if (weights_.size() < 2) return;
    long long common_divisor = weights_[0].integer_mass;
    for (size_t i = 1; i < weights_.size(); ++i) {
        common_divisor = gcd(common_divisor, weights_[i].integer_mass);
    }
    if (common_divisor > 1) {
        for (auto& w : weights_) {
            w.integer_mass /= common_divisor;
        }
    }
}

void MassDecomposer::calc_ert() {
    if (weights_.empty()) return;
    long long a1 = weights_[0].integer_mass;
    if (a1 <= 0) return;
    ert_.assign(weights_.size(), std::vector<long long>(a1, 0));
    for (size_t i = 1; i < weights_.size(); ++i) {
        long long a = weights_[i].integer_mass;
        for (long long r = 1; r < a1; ++r) {
            long long min_val = -1;
            for (long long k = 0; k < a; ++k) {
                long long val = ert_[i-1][(r + k * a1) % a] + (r + k * a1) / a;
                if (min_val == -1 || val < min_val) {
                    min_val = val;
                }
            }
            ert_[i][r] = min_val;
        }
    }
}

void MassDecomposer::compute_errors() {
    min_error_ = 0;
    max_error_ = 0;
    for(const auto& w : weights_) {
        double error = w.integer_mass * precision_ - w.mass;
        min_error_ += (error > 0 ? w.min_count : w.max_count) * error;
        max_error_ += (error > 0 ? w.max_count : w.min_count) * error;
    }
}

long long MassDecomposer::gcd(long long u, long long v) const {
    while (v) {
        long long r = u % v;
        u = v;
        v = r;
    }
    return u;
}

std::pair<long long, long long> MassDecomposer::integer_bound(double mass_from, double mass_to) const {
    return {static_cast<long long>(std::ceil(mass_from / precision_ - max_error_)), static_cast<long long>(std::floor(mass_to / precision_ - min_error_))};
}

bool MassDecomposer::decomposable(int i, long long m, long long a1) const {
    if (a1 <= 0) return false;
    return m >= ert_[i][m % a1];
}

void MassDecomposer::initialize_residue_tables() {
    min_residues_.assign(weights_.size(), 0.0);
    max_residues_.assign(weights_.size(), 0.0);
    for (int i = static_cast<int>(weights_.size()) - 2; i >= 0; --i) {
        min_residues_[i] = min_residues_[i+1] + weights_[i+1].min_count * weights_[i+1].mass;
        max_residues_[i] = max_residues_[i+1] + weights_[i+1].max_count * weights_[i+1].mass;
    }
}

bool MassDecomposer::can_reach_target(double current_mass, int level, double target_mass, double tolerance) const {
    return current_mass + min_residues_[level] <= target_mass + tolerance && current_mass + max_residues_[level] >= target_mass - tolerance;
}