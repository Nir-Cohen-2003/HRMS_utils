#include "mass_decomposer_core.hpp"
#include <omp.h>
#include <climits>
#include <cstdlib>
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

void MassDecomposer::init_money_changing() {
    // Sort elements by mass (smallest first for money-changing)
    auto sorted_elements = elements_;
    std::sort(sorted_elements.begin(), sorted_elements.end(),
              [this](const Element& a, const Element& b) {
                  return atomic_masses_[a.symbol] < atomic_masses_[b.symbol];
              });
    
    weights_.resize(sorted_elements.size());
    for (size_t i = 0; i < sorted_elements.size(); ++i) {
        weights_[i].symbol = sorted_elements[i].symbol;
        weights_[i].mass = atomic_masses_[sorted_elements[i].symbol];
        weights_[i].min_count = sorted_elements[i].min_count;
        weights_[i].max_count = sorted_elements[i].max_count;
        
        // Store indices for constraint elements
        if (weights_[i].symbol == "C") c_idx_ = i;
        else if (weights_[i].symbol == "H") h_idx_ = i;
        else if (weights_[i].symbol == "N") n_idx_ = i;
        else if (weights_[i].symbol == "P") p_idx_ = i;
        else if (weights_[i].symbol == "F") f_idx_ = i;
        else if (weights_[i].symbol == "Cl") cl_idx_ = i;
        else if (weights_[i].symbol == "Br") br_idx_ = i;
        else if (weights_[i].symbol == "I") i_idx_ = i;
    }
}

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

long long MassDecomposer::gcd(long long u, long long v) const {
    while (v != 0) {
        long long r = u % v;
        u = v;
        v = r;
    }
    return u;
}

void MassDecomposer::discretize_masses() {
    for (auto& weight : weights_) {
        weight.integer_mass = static_cast<long long>(weight.mass / precision_);
    }
}

void MassDecomposer::divide_by_gcd() {
    if (weights_.size() < 2) return;
    
    long long d = gcd(weights_[0].integer_mass, weights_[1].integer_mass);
    for (size_t i = 2; i < weights_.size(); ++i) {
        d = gcd(d, weights_[i].integer_mass);
        if (d == 1) break;
    }
    
    if (d > 1) {
        precision_ *= d;
        for (auto& weight : weights_) {
            weight.integer_mass /= d;
        }
    }
}

void MassDecomposer::calc_ert() {
    long long first_long_val = weights_[0].integer_mass;
    if (first_long_val <= 0) {
        throw std::runtime_error("First element mass is zero or negative after discretization.");
    }

    ert_.assign(first_long_val, std::vector<long long>(weights_.size()));
    
    ert_[0][0] = 0;
    for (int i = 1; i < first_long_val; ++i) {
        ert_[i][0] = LLONG_MAX;
    }

    for (size_t j = 1; j < weights_.size(); ++j) {
        ert_[0][j] = 0;
        long long d = gcd(first_long_val, weights_[j].integer_mass);
        
        for (int p = 0; p < d; ++p) {
            long long n = LLONG_MAX;
            for (int i = p; i < first_long_val; i += d) {
                if (ert_[i][j-1] < n) {
                    n = ert_[i][j-1];
                }
            }
            
            if (n == LLONG_MAX) {
                for (int i = p; i < first_long_val; i += d) {
                    ert_[i][j] = LLONG_MAX;
                }
            } else {
                for (int i = 0; i < first_long_val / d; ++i) {
                    n += weights_[j].integer_mass;
                    int r = static_cast<int>(n % first_long_val);
                    if (ert_[r][j-1] < n) {
                        n = ert_[r][j-1];
                    }
                    ert_[r][j] = n;
                }
            }
        }
    }
}

void MassDecomposer::compute_errors() {
    min_error_ = 0.0;
    max_error_ = 0.0;
    for (const auto& weight : weights_) {
        if (weight.mass == 0) continue;
        double error = (precision_ * weight.integer_mass - weight.mass) / weight.mass;
        if (error < min_error_) min_error_ = error;
        if (error > max_error_) max_error_ = error;
    }
}

std::pair<long long, long long> MassDecomposer::integer_bound(double mass_from, double mass_to) const {
    double from_d = std::ceil((1 + min_error_) * mass_from / precision_);
    double to_d = std::floor((1 + max_error_) * mass_to / precision_);
    
    if (from_d > LLONG_MAX || to_d > LLONG_MAX) {
        throw std::runtime_error("Mass too large for 64-bit integer space.");
    }
    
    long long start = static_cast<long long>(std::max(0.0, from_d));
    long long end = static_cast<long long>(std::max(static_cast<double>(start), to_d));
    return {start, end};
}

bool MassDecomposer::decomposable(int i, long long m, long long a1) const {
    if (m < 0) return false;
    return ert_[m % a1][i] <= m;
}

std::vector<Formula> MassDecomposer::integer_decompose(long long mass) const {
    std::vector<Formula> results;
    int k = static_cast<int>(weights_.size()) - 1;
    if (k < 0) return results;
    
    long long a = weights_[0].integer_mass;
    if (a <= 0) return results;

    std::vector<int> c(k + 1, 0);
    int i = k;
    long long m = mass;
    
    while (i <= k) {
        if (!decomposable(i, m, a)) {
            while (i <= k && !decomposable(i, m, a)) {
                m += c[i] * weights_[i].integer_mass;
                c[i] = 0;
                i++;
            }
            
            if (i <= k) {
                m -= weights_[i].integer_mass;
                c[i]++;
            }
        } else {
            while (i > 0 && decomposable(i - 1, m, a)) {
                i--;
            }
            
            if (i == 0) {
                if (a > 0) {
                    c[0] = static_cast<int>(m / a);
                } else {
                    c[0] = 0;
                }

                // Check element bounds
                bool valid_formula = true;
                for (int j = 0; j <= k; ++j) {
                    if (c[j] < weights_[j].min_count || c[j] > weights_[j].max_count) {
                        valid_formula = false;
                        break;
                    }
                }
                
                if (valid_formula) {
                    Formula res;
                    for (int j = 0; j <= k; ++j) {
                        if (c[j] > 0) {
                            res[weights_[j].symbol] = c[j];
                        }
                    }
                    if (!res.empty()) {
                        results.push_back(res);
                    }
                }
                i++;
            }
            
            while (i <= k && c[i] >= weights_[i].max_count) {
                m += c[i] * weights_[i].integer_mass;
                c[i] = 0;
                i++;
            }

            if (i <= k) {
                m -= weights_[i].integer_mass;
                c[i]++;
            }
        }
    }
    
    return results;
}

void MassDecomposer::initialize_residue_tables() {
    min_residues_.resize(elements_.size());
    max_residues_.resize(elements_.size());
    
    double min_mass = 0.0, max_mass = 0.0;
    for (int i = static_cast<int>(elements_.size()) - 1; i >= 0; --i) {
        min_mass += elements_[i].min_count * atomic_masses_[elements_[i].symbol];
        max_mass += elements_[i].max_count * atomic_masses_[elements_[i].symbol];
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

std::vector<std::vector<Formula>> MassDecomposer::decompose_parallel(
    const std::vector<double>& target_masses, 
    const DecompositionParams& params) {
    
    int n_masses = static_cast<int>(target_masses.size());
    std::vector<std::vector<Formula>> all_results(n_masses);
    
    // Use OpenMP for true parallel processing
    #pragma omp parallel
    {
        // Each thread needs its own decomposer instance to avoid race conditions
        MassDecomposer thread_decomposer(elements_, params.strategy);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_masses; ++i) {
            all_results[i] = thread_decomposer.decompose(target_masses[i], params);
        }
    }
    
    return all_results;
}

ProperSpectrumResults MassDecomposer::decompose_spectrum(
    double precursor_mass,
    const std::vector<double>& fragment_masses,
    const DecompositionParams& params) {
    ProperSpectrumResults results;

    // Step 1: Decompose precursor mass to get all possible precursor formulas
    std::vector<Formula> precursor_formulas = decompose(precursor_mass, params);

    // Step 2: For each precursor formula, decompose fragments using it as maximal bounds
    for (const Formula& precursor_formula : precursor_formulas) {
        SpectrumDecomposition decomp;
        decomp.precursor = precursor_formula;

        // Calculate precursor mass and error
        decomp.precursor_mass = 0.0;
        for (const auto& pair : precursor_formula) {
            auto it = atomic_masses_.find(pair.first);
            if (it != atomic_masses_.end()) {
                decomp.precursor_mass += it->second * pair.second;
            }
        }
        decomp.precursor_error_ppm = std::abs(decomp.precursor_mass - precursor_mass) / precursor_mass * 1e6;

        // Create bounds for fragment decomposition based on precursor formula
        std::vector<Element> fragment_bounds = create_bounds_from_formula(precursor_formula);

        // Create a temporary decomposer with fragment bounds
        MassDecomposer fragment_decomposer(fragment_bounds, params.strategy);

        // Decompose each fragment mass
        decomp.fragments.resize(fragment_masses.size());
        decomp.fragment_masses.resize(fragment_masses.size());
        decomp.fragment_errors_ppm.resize(fragment_masses.size());

        for (size_t i = 0; i < fragment_masses.size(); ++i) {
            double fragment_mass = fragment_masses[i];

            // Use relaxed DBE constraints for fragments (they can be lower)
            DecompositionParams fragment_params = params;
            fragment_params.min_dbe = 0.0;  // Fragments can have lower DBE

            std::vector<Formula> fragment_formulas = fragment_decomposer.decompose(fragment_mass, fragment_params);

            // Calculate masses and errors for each fragment formula
            for (const Formula& frag_formula : fragment_formulas) {
                double calc_mass = 0.0;
                for (const auto& pair : frag_formula) {
                    auto it = atomic_masses_.find(pair.first);
                    if (it != atomic_masses_.end()) {
                        calc_mass += it->second * pair.second;
                    }
                }
                double error_ppm = std::abs(calc_mass - fragment_mass) / fragment_mass * 1e6;

                decomp.fragment_masses[i].push_back(calc_mass);
                decomp.fragment_errors_ppm[i].push_back(error_ppm);
            }

            decomp.fragments[i] = std::move(fragment_formulas);
        }

        results.decompositions.push_back(std::move(decomp));
    }

    return results;
}

std::vector<ProperSpectrumResults> MassDecomposer::decompose_spectra_parallel(
    const std::vector<Spectrum>& spectra,
    const DecompositionParams& params) {
    
    int n_spectra = static_cast<int>(spectra.size());
    std::vector<ProperSpectrumResults> all_results(n_spectra);
    
    // Use OpenMP for true parallel processing of spectra
    #pragma omp parallel
    {
        // Each thread needs its own decomposer instance to avoid race conditions
        MassDecomposer thread_decomposer(elements_, params.strategy);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_spectra; ++i) {
            const Spectrum& spectrum = spectra[i];
            all_results[i] = thread_decomposer.decompose_spectrum(
                spectrum.precursor_mass, spectrum.fragment_masses, params);
        }
    }
    
    return all_results;
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
