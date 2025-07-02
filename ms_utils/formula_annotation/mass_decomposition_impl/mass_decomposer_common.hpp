#ifndef MASS_DECOMPOSER_COMMON_HPP
#define MASS_DECOMPOSER_COMMON_HPP

#include <vector>
#include <string>
// #include <unordered_map> // not needed anymore, using std::array for fixed-size Formula arrays
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>

namespace FormulaAnnotation {
    // Centralized Element Definition
    constexpr int NUM_ELEMENTS = 15;

    constexpr std::array<const char*, NUM_ELEMENTS> ELEMENT_SYMBOLS = {
        "H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"
    };

    constexpr std::array<double, NUM_ELEMENTS> ATOMIC_MASSES = {
        1.007825, 11.009305, 12.000000, 14.003074, 15.994915, 18.998403, 
        22.989770, 27.9769265, 30.973762, 31.972071, 34.96885271, 
        38.963707, 74.921596, 78.918338, 126.904468
    };

    enum ElementIndex {
        H = 0, B = 1, C = 2, N = 3, O = 4, F = 5, Na = 6, Si = 7, P = 8, 
        S = 9, Cl = 10, K = 11, As = 12, Br = 13, I = 14
    };

    // New Formula Type
    using Formula = std::array<int, NUM_ELEMENTS>;
}

// Result structure for formulas
using Formula = FormulaAnnotation::Formula;

// Spectrum structure for batch processing
struct Spectrum {
    double precursor_mass;
    std::vector<double> fragment_masses;
};

<<<<<<< HEAD:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_core.hpp
// Spectrum structure with per-item bounds
struct SpectrumWithBounds {
=======
// New structure for spectrum with custom bounds for parallel processing
struct SpectrumWithBounds {
    double precursor_mass;
    std::vector<double> fragment_masses;
    Formula precursor_min_bounds;
    Formula precursor_max_bounds;
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
>>>>>>> split_cpp_code:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_common.hpp
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
    double mass_accuracy_ppm;
    double min_dbe;
    double max_dbe;
    double max_hetero_ratio;
    int max_results;
    Formula min_bounds;
    Formula max_bounds;
};

class MassDecomposer {
private:
<<<<<<< HEAD:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_core.hpp
    std::vector<std::string> element_order_;
    std::unordered_map<std::string, double> atomic_masses_;
    std::vector<double> element_masses_vec_;
=======
    Formula min_bounds_;
    Formula max_bounds_;
>>>>>>> split_cpp_code:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_common.hpp
    
    struct Weight {
        int original_index; // Index in the global ELEMENT_SYMBOLS array
        double mass;
        long long integer_mass;
        int min_count;
        int max_count;
    };
    
    std::vector<Weight> weights_;
    std::vector<std::vector<long long>> ert_;
    double precision_;
    double min_error_, max_error_;
    
<<<<<<< HEAD:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_core.hpp
    int c_idx_, h_idx_, n_idx_, p_idx_, f_idx_, cl_idx_, br_idx_, i_idx_;
    
    void init_atomic_masses();
    void init_elements(const BoundsArray& bounds);
    void init_money_changing();
    void init_recursive();
    bool check_chemical_constraints(const FormulaArray& formula, const DecompositionParams& params) const;
=======
    // Helper methods
    void init_money_changing();
    bool check_dbe(const Formula& formula, double min_dbe, double max_dbe) const;
    bool check_hetero_ratio(const Formula& formula, double max_ratio) const;
>>>>>>> split_cpp_code:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_common.hpp
    long long gcd(long long u, long long v) const;
    void discretize_masses();
    void divide_by_gcd();
    void calc_ert();
    void compute_errors();
    std::pair<long long, long long> integer_bound(double mass_from, double mass_to) const;
    bool decomposable(int i, long long m, long long a1) const;
    std::vector<FormulaArray> integer_decompose(long long mass) const;
    
<<<<<<< HEAD:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_core.hpp
    std::vector<double> min_residues_, max_residues_;
    void initialize_residue_tables();
    bool can_reach_target(double current_mass, int level, double target_mass, double tolerance) const;
    void decompose_recursive_impl(std::vector<int>& formula, double current_mass, 
                                int level, double target_mass, double tolerance,
                                const DecompositionParams& params,
                                std::vector<FormulaArray>& results) const;

public:
    MassDecomposer();
    ~MassDecomposer() = default;

    std::vector<std::string> get_element_order() const { return element_order_; }

    std::vector<FormulaArray> decompose(double target_mass, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<FormulaArray> decompose_mass(double target_mass, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_spectrum(double precursor_mass, const std::vector<double>& fragment_masses, const BoundsArray& bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_spectrum_known_precursor(const FormulaArray& precursor_formula, const std::vector<double>& fragment_masses, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_mass_parallel(const std::vector<double>& target_masses, const BoundsArray& uniform_bounds, const DecompositionParams& params);
    std::vector<std::vector<FormulaArray>> decompose_mass_parallel(const std::vector<double>& target_masses, const std::vector<BoundsArray>& per_mass_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_parallel(const std::vector<Spectrum>& spectra, const BoundsArray& uniform_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_parallel(const std::vector<SpectrumWithBounds>& spectra_with_bounds, const DecompositionParams& params);
    std::vector<std::vector<std::vector<FormulaArray>>> decompose_spectrum_known_precursor_parallel(const std::vector<SpectrumWithKnownPrecursor>& spectra, const DecompositionParams& params);
=======
    bool check_chemical_constraints(const Formula& formula, 
                                  const DecompositionParams& params) const;

public:
    MassDecomposer(const Formula& min_bounds, const Formula& max_bounds);
    ~MassDecomposer() = default;
    
    // Single mass decomposition
    std::vector<Formula> decompose(double target_mass, const DecompositionParams& params);
    
    // Parallel mass decomposition (OpenMP)
    static std::vector<std::vector<Formula>> decompose_parallel(
        const std::vector<double>& target_masses, 
        const DecompositionParams& params);
    
    // New: Parallel mass decomposition with per-mass bounds
    static std::vector<std::vector<Formula>> decompose_masses_parallel_per_bounds(
        const std::vector<double>& target_masses,
        const std::vector<std::pair<Formula, Formula>>& per_mass_bounds,
        const DecompositionParams& params);

    // Proper spectrum decomposition - ensures fragments are subsets of precursors
    ProperSpectrumResults decompose_spectrum(
        double precursor_mass,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Proper parallel spectrum decomposition - processes multiple spectra properly in parallel
    static std::vector<ProperSpectrumResults> decompose_spectra_parallel(
        const std::vector<Spectrum>& spectra,
        const DecompositionParams& params);

    // New: Parallel spectrum decomposition with per-spectrum bounds
    static std::vector<ProperSpectrumResults> decompose_spectra_parallel_per_bounds(
        const std::vector<SpectrumWithBounds>& spectra,
        const DecompositionParams& params);
    
    // Known precursor spectrum decomposition - decomposes fragments with known precursor formula
    std::vector<std::vector<Formula>> decompose_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<double>& fragment_masses,
        const DecompositionParams& params);
    
    // Parallel known precursor spectrum decomposition - processes multiple spectra with different known precursor formulas
    static std::vector<std::vector<std::vector<Formula>>> decompose_spectra_known_precursor_parallel(
        const std::vector<SpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
>>>>>>> split_cpp_code:ms_utils/formula_annotation/mass_decomposition_impl/mass_decomposer_common.hpp
};
#endif // MASS_DECOMPOSER_COMMON_HPP