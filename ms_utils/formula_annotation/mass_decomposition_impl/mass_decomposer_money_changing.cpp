#include "mass_decomposer_common.hpp"
#include <climits>
#include <stdexcept>

void MassDecomposer::init_money_changing() {
    weights_.clear();
    for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
        if (max_bounds_[i] > 0) {
            Weight w;
            w.original_index = i;
            w.mass = FormulaAnnotation::ATOMIC_MASSES[i];
            w.min_count = min_bounds_[i];
            w.max_count = max_bounds_[i];
            weights_.push_back(w);
        }
    }
    // Sort by mass (smallest first for money-changing)
    std::sort(weights_.begin(), weights_.end(), [](const Weight& a, const Weight& b) {
        return a.mass < b.mass;
    });
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
    
    long long d = weights_[0].integer_mass;
    if (weights_.size() > 1) {
        d = gcd(weights_[0].integer_mass, weights_[1].integer_mass);
        for (size_t i = 2; i < weights_.size(); ++i) {
            d = gcd(d, weights_[i].integer_mass);
            if (d == 1) break;
        }
    }
    
    if (d > 1) {
        precision_ *= d;
        for (auto& weight : weights_) {
            weight.integer_mass /= d;
        }
    }
}

void MassDecomposer::calc_ert() {
    if (weights_.empty()) return;
    long long first_long_val = weights_[0].integer_mass;
    if (first_long_val <= 0) {
        throw std::runtime_error("First element mass is zero or negative after discretization.");
    }

    ert_.assign(first_long_val, std::vector<long long>(weights_.size()));
    
    ert_[0][0] = 0;
    for (long long i = 1; i < first_long_val; ++i) {
        ert_[i][0] = LLONG_MAX;
    }

    for (size_t j = 1; j < weights_.size(); ++j) {
        ert_[0][j] = 0;
        long long d = gcd(first_long_val, weights_[j].integer_mass);
        
        for (int p = 0; p < d; ++p) {
            long long n = LLONG_MAX;
            for (long long i = p; i < first_long_val; i += d) {
                if (ert_[i][j-1] < n) {
                    n = ert_[i][j-1];
                }
            }
            
            if (n == LLONG_MAX) {
                for (long long i = p; i < first_long_val; i += d) {
                    ert_[i][j] = LLONG_MAX;
                }
            } else {
                for (long long i = 0; i < first_long_val / d; ++i) {
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
    if (a1 <= 0) return false;
    return ert_[m % a1][i] <= m;
}

inline bool MassDecomposer::decomposable_fast(int i, long long m) const {
    if (m < 0) return false;
    return ert_[m % weights_[0].integer_mass][i] <= m;
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
        if (!decomposable_fast(i, m)) {  
            while (i <= k && !decomposable_fast(i, m)) {
                m += c[i] * weights_[i].integer_mass;
                c[i] = 0;
                i++;
            }
            
            if (i <= k) {
                m -= weights_[i].integer_mass;
                c[i]++;
            }
        } else {
            while (i > 0 && decomposable_fast(i-1, m)) {
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
                    Formula res{}; // Initialize with zeros
                    for (int j = 0; j <= k; ++j) {
                        res[weights_[j].original_index] = c[j];
                    }
                    results.push_back(res);
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