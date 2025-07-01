#ifndef MASS_DECOMPOSER_CORE_HPP
#define MASS_DECOMPOSER_CORE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <memory>

// Element structure - now just stores mass and symbol for ordering
struct Element {
    std::string symbol;
    double mass;
};

// New formula representation as array (15 elements in predefined order)
using FormulaArray = std::vector<int>;

// Bounds structure as 2D vector (min_formula, max_formula)
using BoundsArray = std::vector<std::vector<int>>;  // bounds[0] = min_counts, bounds[1] = max_counts

// Legacy formula representation for backward compatibility
using Formula = std::unordered_map<std::string, int>;

// Spectrum structure for batch processing
struct Spectrum {
    double precursor_mass;
    std::vector<double> fragment_masses;
};

// Spectrum structure with per-item bounds
struct SpectrumWithBounds {
    double precursor_mass;
    std::vector<double> fragment_masses;
    BoundsArray bounds;  // [min_formula, max_formula]
};

// Spectrum structure with known precursor formula (array format)
struct SpectrumWithKnownPrecursor {
    FormulaArray precursor_formula;
    std::vector<double> fragment_masses;
};

// Spectrum structure with known precursor and per-item bounds
struct SpectrumWithKnownPrecursorAndBounds {
    FormulaArray precursor_formula;
    std::vector<double> fragment_masses;
    BoundsArray bounds;  // [min_formula, max_formula]
};

// Legacy spectrum structure with known precursor formula (dict format)
struct SpectrumWithKnownPrecursorLegacy {
    Formula precursor_formula;
    std::vector<double> fragment_masses;
};

// Spectrum results structure (array format)
struct SpectrumResultsArray {
    std::vector<FormulaArray> precursor_results;
    std::vector<std::vector<FormulaArray>> fragment_results;
};

// Proper spectrum results structure where fragments are subsets of precursors (array format)
struct SpectrumDecompositionArray {
    FormulaArray precursor;
    std::vector<std::vector<FormulaArray>> fragments;  // fragments[i] = all possible formulas for fragment mass i
    double precursor_mass;
    double precursor_error_ppm;
    std::vector<std::vector<double>> fragment_masses;    // fragment_masses[i] = masses for fragment i formulas
    std::vector<std::vector<double>> fragment_errors_ppm; // fragment_errors_ppm[i] = errors for fragment i formulas
};

struct ProperSpectrumResultsArray {
    std::vector<SpectrumDecompositionArray> decompositions;
};

// Legacy spectrum results structure (dict format - for backward compatibility)
struct SpectrumResults {
    std::vector<Formula> precursor_results;
    std::vector<std::vector<Formula>> fragment_results;
};

// Legacy proper spectrum results structure where fragments are subsets of precursors (dict format)
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
    std::vector<std::string> element_order_;  // Fixed order of elements
    
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
    void init_money_changing_with_bounds(const BoundsArray& bounds);
    void init_recursive();
    bool check_dbe(const FormulaArray& formula, double min_dbe, double max_dbe) const;
    bool check_dbe_legacy(const Formula& formula, double min_dbe, double max_dbe) const;
    bool check_hetero_ratio(const FormulaArray& formula, double max_ratio) const;
    bool check_hetero_ratio_legacy(const Formula& formula, double max_ratio) const;
    long long gcd(long long u, long long v) const;
    void discretize_masses();
    void divide_by_gcd();
    void calc_ert();
    void compute_errors();
    std::pair<long long, long long> integer_bound(double mass_from, double mass_to) const;
    bool decomposable(int i, long long m, long long a1) const;
    std::vector<FormulaArray> integer_decompose(long long mass) const;
    std::vector<FormulaArray> integer_decompose_with_bounds(long long mass, const BoundsArray& bounds) const;
    
    // Conversion helpers
    FormulaArray formula_to_array(const Formula& formula) const;
    Formula array_to_formula(const FormulaArray& formula_array) const;
    BoundsArray legacy_bounds_to_array(const std::vector<Element>& legacy_bounds) const;
    
    // Recursive algorithm helpers
    std::vector<double> min_residues_, max_residues_;
    void initialize_residue_tables();
    bool can_reach_target(double current_mass, int level, double target_mass, double tolerance) const;
    void decompose_recursive_impl(std::vector<int>& formula, double current_mass, 
                                int level, double target_mass, double tolerance,
                                const DecompositionParams& params,
                                std::vector<FormulaArray>& results) const;
    void decompose_recursive_impl_with_bounds(std::vector<int>& formula, double current_mass, 
                                int level, double target_mass, double tolerance,
                                const DecompositionParams& params, const BoundsArray& bounds,
                                std::vector<FormulaArray>& results) const;
    bool check_chemical_constraints(const std::vector<int>& formula, 
                                  const DecompositionParams& params) const;
    bool check_bounds_constraints(const std::vector<int>& formula, const BoundsArray& bounds) const;

public:
    MassDecomposer(const std::vector<Element>& elements, const std::string& strategy);
    ~MassDecomposer() = default;
    
    // ========== NEW ARRAY-BASED API ==========
    
    // Single mass decomposition (array format)
    std::vector<FormulaArray> decompose_array(double target_mass, const BoundsArray& bounds, const DecompositionParams& params);
    
    // Parallel mass decomposition with uniform bounds (array format)
    std::vector<std::vector<FormulaArray>> decompose_parallel_array(
        const std::vector<double>& target_masses, 
        const BoundsArray& bounds,
        const DecompositionParams& params);
    
    // Parallel mass decomposition with per-mass bounds (array format)
    std::vector<std::vector<FormulaArray>> decompose_parallel_with_bounds_array(
        const std::vector<double>& target_masses, 
        const std::vector<BoundsArray>& bounds_per_mass,
        const DecompositionParams& params);
    
    // Spectrum decomposition with uniform bounds (array format)
    std::vector<std::vector<FormulaArray>> decompose_spectrum_array(
        double precursor_mass,
        const std::vector<double>& fragment_masses,
        const BoundsArray& bounds,
        const DecompositionParams& params);
    
    // Parallel spectrum decomposition with uniform bounds (array format)
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectra_parallel_array(
        const std::vector<Spectrum>& spectra,
        const BoundsArray& bounds,
        const DecompositionParams& params);
    
    // Parallel spectrum decomposition with per-spectrum bounds (array format)
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectra_parallel_with_bounds_array(
        const std::vector<SpectrumWithBounds>& spectra,
        const DecompositionParams& params);
    
    // Known precursor spectrum decomposition (array format)
    std::vector<std::vector<FormulaArray>> decompose_spectrum_known_precursor_array(
        const FormulaArray& precursor_formula,
        const std::vector<double>& fragment_masses,
        const BoundsArray& bounds,
        const DecompositionParams& params);
    
    // Parallel known precursor spectrum decomposition with uniform bounds (array format)
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectra_known_precursor_parallel_array(
        const std::vector<SpectrumWithKnownPrecursor>& spectra,
        const BoundsArray& bounds,
        const DecompositionParams& params);
    
    // Parallel known precursor spectrum decomposition with per-spectrum bounds (array format)
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectra_known_precursor_parallel_with_bounds_array(
        const std::vector<SpectrumWithKnownPrecursorAndBounds>& spectra,
        const DecompositionParams& params);
    
    // ========== LEGACY DICT-BASED API ==========
    
    // Single mass decomposition (legacy dict format)
    std::vector<Formula> decompose(double target_mass, const DecompositionParams& params);
    
    // Parallel mass decomposition (legacy dict format)
    std::vector<std::vector<Formula>> decompose_parallel(
        const std::vector<double>& target_masses, 
        const DecompositionParams& params);
    
    // Proper spectrum decomposition - ensures fragments are subsets of precursors (legacy dict format)
    ProperSpectrumResults decompose_spectrum(
        double precursor_mass,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Proper parallel spectrum decomposition - processes multiple spectra properly in parallel (legacy dict format)
    std::vector<ProperSpectrumResults> decompose_spectra_parallel(
        const std::vector<Spectrum>& spectra,
        const DecompositionParams& params);
    
    // Known precursor spectrum decomposition - decomposes fragments with known precursor formula (legacy dict format)
    std::vector<std::vector<Formula>> decompose_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Parallel known precursor spectrum decomposition - processes multiple spectra with different known precursor formulas (legacy dict format)
    std::vector<std::vector<std::vector<Formula>>> decompose_spectra_known_precursor_parallel(
        const std::vector<SpectrumWithKnownPrecursorLegacy>& spectra,
        const DecompositionParams& params);
    
    // Helper function to create element bounds from a formula (for fragment decomposition) (legacy)
    std::vector<Element> create_bounds_from_formula(const Formula& formula) const;
    
    // Helper function to create bounds array from a formula (for fragment decomposition)
    BoundsArray create_bounds_array_from_formula(const FormulaArray& formula) const;
    
    // Helper function for Python interface (legacy)
    std::vector<std::pair<std::string, int>> formula_to_pairs(const Formula& formula) const;
    
    // Get element order
    const std::vector<std::string>& get_element_order() const { return element_order_; }
};

#endif // MASS_DECOMPOSER_CORE_HPP
