#include "mass_decomposer_common.hpp"
#include <climits>
#include <stdexcept>

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