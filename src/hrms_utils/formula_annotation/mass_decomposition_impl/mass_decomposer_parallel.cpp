#include "mass_decomposer_common.hpp"
#include <omp.h>

std::vector<std::vector<Formula>> MassDecomposer::decompose_parallel(
    const std::vector<double>& target_masses, 
    const DecompositionParams& params) {
    
    int n_masses = static_cast<int>(target_masses.size());
    std::vector<std::vector<Formula>> all_results(n_masses);
    
    #pragma omp parallel
    {
        MassDecomposer thread_decomposer(params.min_bounds, params.max_bounds);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_masses; ++i) {
            all_results[i] = thread_decomposer.decompose(target_masses[i], params);
        }
    }
    
    return all_results;
}

std::vector<std::vector<Formula>> MassDecomposer::decompose_masses_parallel_per_bounds(
    const std::vector<double>& target_masses,
    const std::vector<std::pair<Formula, Formula>>& per_mass_bounds,
    const DecompositionParams& params) {

    int n_masses = static_cast<int>(target_masses.size());
    std::vector<std::vector<Formula>> all_results(n_masses);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_masses; ++i) {
        MassDecomposer thread_decomposer(per_mass_bounds[i].first, per_mass_bounds[i].second);
        all_results[i] = thread_decomposer.decompose(target_masses[i], params);
    }

    return all_results;
}

ProperSpectrumResults MassDecomposer::decompose_spectrum(
    double precursor_mass,
    const std::vector<double>& fragment_masses,
    const DecompositionParams& params) {
    ProperSpectrumResults results;

    // Decompose precursor mass
    std::vector<Formula> precursor_formulas = decompose(precursor_mass, params);

    // For each precursor formula reuse existing helper
    for (const Formula& precursor_formula : precursor_formulas) {
        SpectrumDecomposition decomp;
        decomp.precursor = precursor_formula;

        // Calculate precursor mass and absolute error
        decomp.precursor_mass = 0.0;
        for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
            decomp.precursor_mass += precursor_formula[i] * FormulaAnnotation::ATOMIC_MASSES[i];
        }
        decomp.precursor_error_ppm = std::abs(decomp.precursor_mass - precursor_mass);

        // Get fragment decompositions (unfiltered) using existing function
        auto fragment_solutions = decompose_spectrum_known_precursor(
            precursor_formula, fragment_masses, params);

        decomp.fragments = fragment_solutions;
        decomp.fragment_masses.resize(fragment_solutions.size());
        decomp.fragment_errors_ppm.resize(fragment_solutions.size());

        // Populate masses & errors with filtering (same logic as before)
        for (size_t j = 0; j < fragment_solutions.size(); ++j) {
            double target_mass = fragment_masses[j];
            // double allowed_error = std::max(target_mass, 200.0) * params.tolerance_ppm / 1e6;

            for (const auto& frag_formula : fragment_solutions[j]) {
                double calc_mass = 0.0;
                for (int k = 0; k < FormulaAnnotation::NUM_ELEMENTS; ++k) {
                    calc_mass += frag_formula[k] * FormulaAnnotation::ATOMIC_MASSES[k];
                }
                double abs_error = std::abs(calc_mass - target_mass);
                // if (abs_error > allowed_error) continue;
                decomp.fragment_masses[j].push_back(calc_mass);
                decomp.fragment_errors_ppm[j].push_back(abs_error);
            }
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
    
    #pragma omp parallel
    {
        MassDecomposer thread_decomposer(params.min_bounds, params.max_bounds);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_spectra; ++i) {
            const Spectrum& spectrum = spectra[i];
            all_results[i] = thread_decomposer.decompose_spectrum(
                spectrum.precursor_mass, spectrum.fragment_masses, params);
        }
    }
    
    return all_results;
}

std::vector<ProperSpectrumResults> MassDecomposer::decompose_spectra_parallel_per_bounds(
    const std::vector<SpectrumWithBounds>& spectra,
    const DecompositionParams& params) {
    
    int n_spectra = static_cast<int>(spectra.size());
    std::vector<ProperSpectrumResults> all_results(n_spectra);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_spectra; ++i) {
        const auto& spectrum = spectra[i];
        MassDecomposer thread_decomposer(spectrum.precursor_min_bounds, spectrum.precursor_max_bounds);
        all_results[i] = thread_decomposer.decompose_spectrum(
            spectrum.precursor_mass, spectrum.fragment_masses, params);
    }
    
    return all_results;
}

std::vector<std::vector<Formula>> MassDecomposer::decompose_spectrum_known_precursor(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const DecompositionParams& params) {
    
    std::vector<std::vector<Formula>> fragment_results;
    fragment_results.resize(fragment_masses.size());
    
    Formula fragment_min_bounds{};
    Formula fragment_max_bounds = precursor_formula;
    
    MassDecomposer fragment_decomposer(fragment_min_bounds, fragment_max_bounds);
    DecompositionParams fragment_params = params;

    
    for (size_t j = 0; j < fragment_masses.size(); ++j) {
        fragment_results[j] = fragment_decomposer.decompose(fragment_masses[j], fragment_params);
    }
    
    return fragment_results;
}

std::vector<std::vector<std::vector<Formula>>> MassDecomposer::decompose_spectra_known_precursor_parallel(
    const std::vector<SpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {
    
    int n_spectra = static_cast<int>(spectra.size());
    std::vector<std::vector<std::vector<Formula>>> all_results(n_spectra);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_spectra; ++i) {
        const SpectrumWithKnownPrecursor& spectrum = spectra[i];
        
        Formula fragment_min_bounds{};
        Formula fragment_max_bounds = spectrum.precursor_formula;
        MassDecomposer thread_decomposer(fragment_min_bounds, fragment_max_bounds);

        all_results[i] = thread_decomposer.decompose_spectrum_known_precursor(
            spectrum.precursor_formula, spectrum.fragment_masses, params);
    }
    
    return all_results;
}

MassDecomposer::CleanedSpectrumResult MassDecomposer::clean_spectrum_known_precursor(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const std::vector<double>& fragment_intensities,
    const DecompositionParams& params) {

    MassDecomposer::CleanedSpectrumResult out;

    // Compute fragment solutions constrained by precursor formula
    auto fragment_solutions = decompose_spectrum_known_precursor(precursor_formula, fragment_masses, params);

    // Sanity: masses and intensities size must match solutions size
    const size_t n = std::min(fragment_masses.size(), fragment_intensities.size());
    if (fragment_solutions.size() != n) {
        // Truncate safely to the minimum observed size
    }

    out.masses.reserve(n);
    out.intensities.reserve(n);
    out.fragment_formulas.reserve(n);
    out.fragment_errors_ppm.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const double target = fragment_masses[i];
        const double denom_allowed = std::max(target, 200.0);            // filtering
        const double allowed_abs = denom_allowed * params.tolerance_ppm / 1e6;

        const auto& formulas = fragment_solutions[i];
        if (formulas.empty()) {
            continue; // drop fragment with no formulas
        }

        std::vector<double> errors_ppm;
        errors_ppm.reserve(formulas.size());

        // Recompute error for reporting in ppm using the actual target mass
        const double denom_report = (target != 0.0) ? target : denom_allowed;

        for (const auto& f : formulas) {
            double calc_mass = 0.0;
            for (int k = 0; k < FormulaAnnotation::NUM_ELEMENTS; ++k) {
                calc_mass += f[k] * FormulaAnnotation::ATOMIC_MASSES[k];
            }
            const double abs_err = std::abs(calc_mass - target);
            if (abs_err > allowed_abs) {
                continue; // filtered by tolerance
            }
            const double ppm = abs_err * 1e6 / denom_report; // report relative to actual mass
            errors_ppm.push_back(ppm);
        }

        if (errors_ppm.empty()) {
            continue; // all formulas filtered by tolerance => drop fragment
        }

        out.masses.push_back(target);
        out.intensities.push_back(fragment_intensities[i]);
        out.fragment_formulas.push_back(formulas);
        out.fragment_errors_ppm.push_back(std::move(errors_ppm));
    }

    return out;
}

std::vector<MassDecomposer::CleanedSpectrumResult> MassDecomposer::clean_spectra_known_precursor_parallel(
    const std::vector<MassDecomposer::CleanSpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {

    const int n = static_cast<int>(spectra.size());
    std::vector<MassDecomposer::CleanedSpectrumResult> all_results(n);

    #pragma omp parallel
    {
        // Thread-local decomposer instance to call non-static member
        MassDecomposer thread_decomposer(params.min_bounds, params.max_bounds);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            const auto& s = spectra[i];
            all_results[i] = thread_decomposer.clean_spectrum_known_precursor(
                s.precursor_formula, s.fragment_masses, s.fragment_intensities, params);
        }
    }
    return all_results;
}