#ifndef MASS_DECOMPOSER_COMMON_HPP
#define MASS_DECOMPOSER_COMMON_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>
#include <cstdint>
#include <cstddef>

// Centralized floating point type for masses and intensities
using mass_t = double;

namespace FormulaAnnotation {
    // Centralized Element Definition
    constexpr int NUM_ELEMENTS = 15;

    constexpr std::array<const char*, NUM_ELEMENTS> ELEMENT_SYMBOLS = {
        "H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"
    };

    constexpr std::array<mass_t, NUM_ELEMENTS> ATOMIC_MASSES = {
        1.007825, 11.009305, 12.000000, 14.003074, 15.994915, 18.998403, 
        22.989770, 27.9769265, 30.973762, 31.972071, 34.96885271, 
        38.963707, 74.921596, 78.918338, 126.904468
    };

    enum ElementIndex {
        H = 0, B = 1, C = 2, N = 3, O = 4, F = 5, Na = 6, Si = 7, P = 8, 
        S = 9, Cl = 10, K = 11, As = 12, Br = 13, I = 14
    };

    // New Formula Type
    using Formula = std::array<int32_t, NUM_ELEMENTS>;

    // Inline accessors so Cython can call functions instead of linking to constexpr objects.
    inline const char* element_symbol_at(int i) { return ELEMENT_SYMBOLS[i]; }
    inline mass_t atomic_mass_at(int i) { return ATOMIC_MASSES[i]; }

    // Size helpers for safe bulk copies
    inline constexpr std::size_t FORMULA_NBYTES() {
        return sizeof(int32_t) * static_cast<std::size_t>(NUM_ELEMENTS);
    }
    inline const int32_t* formula_data_const(const Formula& f) noexcept {
        return f.data();
    }
    inline int32_t* formula_data(Formula& f) noexcept {
        return f.data();
    }
}

// Result structure for formulas
using Formula = FormulaAnnotation::Formula;

struct FormulaWithString {
    Formula formula;
    std::string formula_string;
};

// Spectrum structure for batch processing
struct Spectrum {
    mass_t precursor_mass;
    std::vector<mass_t> fragment_masses;
};

// New structure for spectrum with custom bounds for parallel processing
struct SpectrumWithBounds {
    mass_t precursor_mass;
    std::vector<mass_t> fragment_masses;
    Formula precursor_min_bounds;
    Formula precursor_max_bounds;
};

// Spectrum structure with known precursor formula
struct SpectrumWithKnownPrecursor {
    Formula precursor_formula;
    std::vector<mass_t> fragment_masses;
};

// Proper spectrum results structure where fragments are subsets of precursors
struct SpectrumDecomposition {
    Formula precursor;
    std::vector<std::vector<Formula>> fragments;  // fragments[i] = all possible formulas for fragment mass i
    mass_t precursor_mass;
    mass_t precursor_error_ppm;
    std::vector<std::vector<mass_t>> fragment_masses;    // fragment_masses[i] = masses for fragment i formulas
    std::vector<std::vector<mass_t>> fragment_errors_ppm; // fragment_errors_ppm[i] = errors for fragment i formulas
};

struct ProperSpectrumResults {
    std::vector<SpectrumDecomposition> decompositions;
};

struct SpectrumDecompositionVerbose {
    Formula precursor;
    std::string precursor_string;
    std::vector<std::vector<Formula>> fragments;
    std::vector<std::vector<std::string>> fragment_strings;
    mass_t precursor_mass;
    mass_t precursor_error_ppm;
    std::vector<std::vector<mass_t>> fragment_masses;
    std::vector<std::vector<mass_t>> fragment_errors_ppm;
};

struct ProperSpectrumResultsVerbose {
    std::vector<SpectrumDecompositionVerbose> decompositions;
};

// Parameters structure for decomposition
struct DecompositionParams {
    mass_t tolerance_ppm;
    mass_t min_dbe;
    mass_t max_dbe;
    int dbe_mode; // 0: integer, 1: half-integer, 2: any
    // mass_t max_hetero_ratio;
    Formula min_bounds;
    Formula max_bounds;
};

// Main decomposer class
class MassDecomposer {
private:
    Formula min_bounds_;
    Formula max_bounds_;
    
    // For money-changing algorithm
    struct Weight {
        int original_index; // Index in the global ELEMENT_SYMBOLS array
        mass_t mass;
        long long integer_mass;
        int min_count;
        int max_count;
    };
    mutable std::vector<int> temp_counts_;
    std::vector<Weight> weights_;
    std::vector<std::vector<long long>> ert_;
    mass_t precision_;
    mass_t min_error_, max_error_;
    bool is_initialized_;
    
    // Helper methods
    void init_money_changing();
    inline bool check_dbe(const Formula& formula, mass_t min_dbe, mass_t max_dbe, int dbe_mode) const;
    // bool check_hetero_ratio(const Formula& formula, mass_t max_ratio) const;
    long long gcd(long long u, long long v) const;
    void discretize_masses();
    void divide_by_gcd();
    void calc_ert();
    void compute_errors();
    std::pair<long long, long long> integer_bound(mass_t mass_from, mass_t mass_to) const;
    bool decomposable(int i, long long m, long long a1) const;
    inline bool decomposable_fast(int i, long long m, long long remainder) const; // Fast check for decomposability
    std::vector<Formula> integer_decompose(long long mass) const;
    
public:
    MassDecomposer(const Formula& min_bounds, const Formula& max_bounds);
    ~MassDecomposer() = default;
    
    static std::string formula_to_string(const Formula& formula);

    // Single mass decomposition
    std::vector<Formula> decompose(mass_t target_mass, const DecompositionParams& params);
    std::vector<FormulaWithString> decompose_verbose(mass_t target_mass, const DecompositionParams& params);
    
    // Parallel mass decomposition (OpenMP)
    static std::vector<std::vector<Formula>> decompose_parallel(
        const std::vector<mass_t>& target_masses, 
        const DecompositionParams& params);
    static std::vector<std::vector<FormulaWithString>> decompose_parallel_verbose(
        const std::vector<mass_t>& target_masses,
        const DecompositionParams& params);
    
    // New: Parallel mass decomposition with per-mass bounds
    static std::vector<std::vector<Formula>> decompose_masses_parallel_per_bounds(
        const std::vector<mass_t>& target_masses,
        const std::vector<std::pair<Formula, Formula>>& per_mass_bounds,
        const DecompositionParams& params);
    static std::vector<std::vector<FormulaWithString>> decompose_masses_parallel_per_bounds_verbose(
        const std::vector<mass_t>& target_masses,
        const std::vector<std::pair<Formula, Formula>>& per_mass_bounds,
        const DecompositionParams& params);

    // Proper spectrum decomposition - ensures fragments are subsets of precursors
    ProperSpectrumResults decompose_spectrum(
        mass_t precursor_mass,
        const std::vector<mass_t>& fragment_masses,
        const DecompositionParams& params);
    ProperSpectrumResultsVerbose decompose_spectrum_verbose(
        mass_t precursor_mass,
        const std::vector<mass_t>& fragment_masses,
        const DecompositionParams& params);
    
    // Proper parallel spectrum decomposition - processes multiple spectra properly in parallel
    static std::vector<ProperSpectrumResults> decompose_spectra_parallel(
        const std::vector<Spectrum>& spectra,
        const DecompositionParams& params);
    static std::vector<ProperSpectrumResultsVerbose> decompose_spectra_parallel_verbose(
        const std::vector<Spectrum>& spectra,
        const DecompositionParams& params);

    // New: Parallel spectrum decomposition with per-spectrum bounds
    static std::vector<ProperSpectrumResults> decompose_spectra_parallel_per_bounds(
        const std::vector<SpectrumWithBounds>& spectra,
        const DecompositionParams& params);
    static std::vector<ProperSpectrumResultsVerbose> decompose_spectra_parallel_per_bounds_verbose(
        const std::vector<SpectrumWithBounds>& spectra,
        const DecompositionParams& params);
    
    // Known precursor spectrum decomposition - decomposes fragments with known precursor formula
    std::vector<std::vector<Formula>> decompose_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const DecompositionParams& params);
    std::vector<std::vector<FormulaWithString>> decompose_spectrum_known_precursor_verbose(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const DecompositionParams& params);

    // Parallel known precursor spectrum decomposition - processes multiple spectra with different known precursor formulas
    static std::vector<std::vector<std::vector<Formula>>> decompose_spectra_known_precursor_parallel(
        const std::vector<SpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
    static std::vector<std::vector<std::vector<FormulaWithString>>> decompose_spectra_known_precursor_parallel_verbose(
        const std::vector<SpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);

    // New: input struct for cleaning with intensities
    struct CleanSpectrumWithKnownPrecursor {
        Formula precursor_formula;
        std::vector<mass_t> fragment_masses;
        std::vector<mass_t> fragment_intensities;
        // Remove precursor_mass field - no longer used for normalization
        mass_t max_allowed_normalized_mass_error_ppm;
    };

    // New: result struct for cleaned spectrum
    struct CleanedSpectrumResult {
        std::vector<mass_t> masses;   // kept fragment masses
        std::vector<mass_t> intensities; // kept fragment intensities
        std::vector<std::vector<Formula>> fragment_formulas;     // per kept fragment
        std::vector<std::vector<mass_t>> fragment_errors_ppm;    // per kept fragment (aligned with formulas)
    };
    struct CleanedSpectrumResultVerbose {
        std::vector<mass_t> masses;
        std::vector<mass_t> intensities;
        std::vector<std::vector<Formula>> fragment_formulas;
        std::vector<std::vector<std::string>> fragment_formulas_strings;
        std::vector<std::vector<mass_t>> fragment_errors_ppm;
    };

    // New: clean a single spectrum with known precursor formula
    CleanedSpectrumResult clean_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const std::vector<mass_t>& fragment_intensities,
        const DecompositionParams& params);
    CleanedSpectrumResultVerbose clean_spectrum_known_precursor_verbose(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const std::vector<mass_t>& fragment_intensities,
        const DecompositionParams& params);

    // New: parallel cleaner for many spectra with known precursor formula
    static std::vector<CleanedSpectrumResult> clean_spectra_known_precursor_parallel(
        const std::vector<CleanSpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
    static std::vector<CleanedSpectrumResultVerbose> clean_spectra_known_precursor_parallel_verbose(
        const std::vector<CleanSpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);

    struct CleanedAndNormalizedSpectrumResult {
        std::vector<mass_t> masses_normalized;          // one per kept fragment (target + final_mean_error)
        std::vector<mass_t> intensities;                // aligned with masses_normalized
        std::vector<Formula> fragment_formulas;         // one per kept fragment
        std::vector<mass_t> fragment_errors_ppm;        // one per kept fragment (after normalization)
    }; 
    struct CleanedAndNormalizedSpectrumResultVerbose {
        std::vector<mass_t> masses_normalized;
        std::vector<mass_t> intensities;
        std::vector<Formula> fragment_formulas;
        std::vector<std::string> fragment_formulas_strings;
        std::vector<mass_t> fragment_errors_ppm;
    };
    CleanedAndNormalizedSpectrumResult clean_and_normalize_spectrum_known_precursor(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const std::vector<mass_t>& fragment_intensities,
        mass_t max_allowed_normalized_mass_error_ppm,
        const DecompositionParams& params);
    CleanedAndNormalizedSpectrumResultVerbose clean_and_normalize_spectrum_known_precursor_verbose(
        const Formula& precursor_formula,
        const std::vector<mass_t>& fragment_masses,
        const std::vector<mass_t>& fragment_intensities,
        mass_t max_allowed_normalized_mass_error_ppm,
        const DecompositionParams& params);

    // New: parallel cleaner + normalizer for many spectra
    static std::vector<CleanedAndNormalizedSpectrumResult> clean_and_normalize_spectra_known_precursor_parallel(
        const std::vector<CleanSpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
    static std::vector<CleanedAndNormalizedSpectrumResultVerbose> clean_and_normalize_spectra_known_precursor_parallel_verbose(
        const std::vector<CleanSpectrumWithKnownPrecursor>& spectra,
        const DecompositionParams& params);
};
#endif // MASS_DECOMPOSER_COMMON_HPP