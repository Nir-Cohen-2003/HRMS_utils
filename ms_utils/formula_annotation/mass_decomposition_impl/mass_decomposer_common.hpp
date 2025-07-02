#ifndef MASS_DECOMPOSER_COMMON_HPP
#define MASS_DECOMPOSER_COMMON_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>

// Element structure
struct Element {
    std::string symbol;
    double mass;
    int min_count;
    int max_count;
};

// Result structure for formulas
using Formula = std::unordered_map<std::string, int>;

// Spectrum structure for batch processing
struct Spectrum {
    double precursor_mass;
    std::vector<double> fragment_masses;
};

// New structure for spectrum with custom bounds for parallel processing
struct SpectrumWithBounds {
    double precursor_mass;
    std::vector<double> fragment_masses;
    std::vector<Element> precursor_bounds;
};

// Spectrum structure with known precursor formula
struct SpectrumWithKnownPrecursor {
    Formula precursor_formula;
    std::vector<double> fragment_masses;
};

// Proper spectrum results structure where fragments are subsets of precursors
struct SpectrumDecomposition {
    Formula precursor;
    std::vector<std::vector<Formula>> fragments;  // fragments[i] = all possible formulas for fragment mass i
    double precursor_mass;
    double precursor_error_ppm;
    std::vector<std::vector<double>> fragment_masses;    // fragment_masses[i] = masses for fragment i formulas
    std::vector<std::vector<double>> fragment_errors_ppm; // fragment_errors_ppm[i] = errors for fragment i formulas
};

struct ProperSpectrumResults {
    std::vector<SpectrumDecomposition> decompositions;
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

// Main decomposer class
class MassDecomposer {
private:
    std::vector<Element> elements_;
    std::unordered_map<std::string, double> atomic_masses_;
    
    // For money-changing algorithm
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
    bool is_initialized_;
    
    // Element indices for constraints
    int c_idx_, h_idx_, n_idx_, p_idx_, f_idx_, cl_idx_, br_idx_, i_idx_;
    
    // Helper methods
    void init_atomic_masses();
    void init_money_changing();
    void init_recursive();
    bool check_dbe(const Formula& formula, double min_dbe, double max_dbe) const;
    bool check_hetero_ratio(const Formula& formula, double max_ratio) const;
    long long gcd(long long u, long long v) const;
    void discretize_masses();
    void divide_by_gcd();
    void calc_ert();
    void compute_errors();
    std::pair<long long, long long> integer_bound(double mass_from, double mass_to) const;
    bool decomposable(int i, long long m, long long a1) const;
    std::vector<Formula> integer_decompose(long long mass) const;
    
    // Recursive algorithm helpers
    std::vector<double> min_residues_, max_residues_;
    void initialize_residue_tables();
    bool can_reach_target(double current_mass, int level, double target_mass, double tolerance) const;
    void decompose_recursive_impl(std::vector<int>& formula, double current_mass, 
                                int level, double target_mass, double tolerance,
                                const DecompositionParams& params,
                                std::vector<Formula>& results) const;
    bool check_chemical_constraints(const std::vector<int>& formula, 
                                  const DecompositionParams& params) const;

public:
    MassDecomposer(const std::vector<Element>& elements, const std::string& strategy);
    ~MassDecomposer() = default;
    
    // Single mass decomposition
    std::vector<Formula> decompose(double target_mass, const DecompositionParams& params);
    
    // Parallel mass decomposition (OpenMP)
    std::vector<std::vector<Formula>> decompose_parallel(
        const std::vector<double>& target_masses, 
        const DecompositionParams& params);
    
    // New: Parallel mass decomposition with per-mass bounds
    std::vector<std::vector<Formula>> decompose_masses_parallel_per_bounds(
        const std::vector<double>& target_masses,
        const std::vector<std::vector<Element>>& per_mass_bounds,
        const DecompositionParams& params);

    // Proper spectrum decomposition - ensures fragments are subsets of precursors
    ProperSpectrumResults decompose_spectrum(
        double precursor_mass,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Proper parallel spectrum decomposition - processes multiple spectra properly in parallel
    std::vector<ProperSpectrumResults> decompose_spectra_parallel(
        const std::vector<Spectrum>& spectra,
        const DecompositionParams& params);

    // New: Parallel spectrum decomposition with per-spectrum bounds
    std::vector<ProperSpectrumResults> decompose_spectra_parallel_per_bounds(
        const std::vector<SpectrumWithBounds>& spectra,
        const DecompositionParams& params);
    
    // Known precursor spectrum decomposition - decomposes fragments with known precursor formula
    std::vector<std::vector<Formula>> decompose_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Parallel known precursor spectrum decomposition - processes multiple spectra with different known precursor formulas
    std::vector<std::vector<std::vector<Formula>>> decompose_spectra_known_precursor_parallel(
        const std::vector<SpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
    
    // Helper function to create element bounds from a formula (for fragment decomposition)
    std::vector<Element> create_bounds_from_formula(const Formula& formula) const;
    
    // Helper function for Python interface
    std::vector<std::pair<std::string, int>> formula_to_pairs(const Formula& formula) const;
};
#endif // MASS_DECOMPOSER_COMMON_HPP