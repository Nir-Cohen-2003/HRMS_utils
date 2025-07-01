#ifndef MASS_DECOMPOSER_CORE_HPP
#define MASS_DECOMPOSER_CORE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>

// Formula representation: fixed-length vector<int> in element_order_
using FormulaArray = std::vector<int>;
// Bounds: 2D vector (min_formula, max_formula), both are FormulaArray
using BoundsArray = std::vector<std::vector<int>>;  // bounds[0] = min_counts, bounds[1] = max_counts

// Spectrum structure for batch processing
struct Spectrum {
    double precursor_mass;
    std::vector<double> fragment_masses;
};

// Spectrum structure with per-item bounds
struct SpectrumWithBounds {
    double precursor_mass;
    std::vector<double> fragment_masses;
    BoundsArray bounds;
};

// Spectrum structure with known precursor formula (array format)
struct SpectrumWithKnownPrecursor {
    FormulaArray precursor_formula;
    std::vector<double> fragment_masses;
};

// Parameters structure for decomposition
struct DecompositionParams {
    double tolerance_ppm;
    double min_dbe;
    double max_dbe;
    double max_hetero_ratio;
    int max_results;
    std::string strategy;
};

class MassDecomposer {
private:
    std::vector<std::string> element_order_;
    std::unordered_map<std::string, double> atomic_masses_;
    std::vector<double> element_masses_vec_;
    
    struct Weight {
        std::string symbol;
        double mass;
        long long integer_mass;
        int min_count;
        int max_count;
    };
    
    std::vector<Weight> weights_;
    std::vector<std::vector<long long>> ert_;
    double precision_;
    double min_error_, max_error_;
    
    int c_idx_, h_idx_, n_idx_, p_idx_, f_idx_, cl_idx_, br_idx_, i_idx_;
    
    void init_atomic_masses();
    void init_elements(const BoundsArray& bounds);
    void init_money_changing();
    void init_recursive();
    bool check_chemical_constraints(const FormulaArray& formula, const DecompositionParams& params) const;
    long long gcd(long long u, long long v) const;
    void discretize_masses();
    void divide_by_gcd();
    void calc_ert();
    void compute_errors();
    std::pair<long long, long long> integer_bound(double mass_from, double mass_to) const;
    bool decomposable(int i, long long m, long long a1) const;
    std::vector<FormulaArray> integer_decompose(long long mass) const;
    
    std::vector<double> min_residues_, max_residues_;
    void initialize_residue_tables();
    bool can_reach_target(double current_mass, int level, double target_mass, double tolerance) const;
    void decompose_recursive_impl(std::vector<int>& formula, double current_mass, 
                                int level, double target_mass, double tolerance,
                                const DecompositionParams& params,
                                std::vector<FormulaArray>& results) const;

public:
    MassDecomposer(const std::vector<std::string>& element_order);
    ~MassDecomposer() = default;

    std::vector<FormulaArray> decompose(double target_mass, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<FormulaArray> decompose_mass(double target_mass, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_spectrum(double precursor_mass, const std::vector<double>& fragment_masses, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_spectrum_known_precursor(const FormulaArray& precursor_formula, const std::vector<double>& fragment_masses, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_mass_parallel(const std::vector<double>& target_masses, const BoundsArray& uniform_bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_mass_parallel(const std::vector<double>& target_masses, const std::vector<BoundsArray>& per_mass_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_parallel(const std::vector<Spectrum>& spectra, const BoundsArray& uniform_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_parallel(const std::vector<SpectrumWithBounds>& spectra_with_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_known_precursor_parallel(const std::vector<SpectrumWithKnownPrecursor>& spectra, const DecompositionParams& params);

    // Corrected function signature
    std::vector<std::string> get_element_order() const;
};

#endif // MASS_DECOMPOSER_CORE_HPP
