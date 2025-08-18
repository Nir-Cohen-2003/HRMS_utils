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

    // Step 1: Decompose precursor mass to get all possible precursor formulas
    std::vector<Formula> precursor_formulas = decompose(precursor_mass, params);

    // Step 2: For each precursor formula, decompose fragments using it as maximal bounds
    for (const Formula& precursor_formula : precursor_formulas) {
        SpectrumDecomposition decomp;
        decomp.precursor = precursor_formula;

        // Calculate precursor mass and error
        decomp.precursor_mass = 0.0;
        for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
            decomp.precursor_mass += precursor_formula[i] * FormulaAnnotation::ATOMIC_MASSES[i];
        }
        // Precursor error now absolute
        decomp.precursor_error_ppm = std::abs(decomp.precursor_mass - precursor_mass);

        // Create bounds for fragment decomposition based on precursor formula
        Formula fragment_min_bounds{}; // All zeros
        Formula fragment_max_bounds = precursor_formula; // Max is what's in the precursor

        MassDecomposer fragment_decomposer(fragment_min_bounds, fragment_max_bounds);
        DecompositionParams fragment_params = params;
        fragment_params.min_dbe = -100.0;  // relaxed DBE for fragments
        fragment_params.max_dbe = 100.0;
        
        decomp.fragments.resize(fragment_masses.size());
        decomp.fragment_masses.resize(fragment_masses.size());
        decomp.fragment_errors_ppm.resize(fragment_masses.size());

        for (size_t j = 0; j < fragment_masses.size(); ++j) {
            auto fragment_solutions = fragment_decomposer.decompose(fragment_masses[j], fragment_params);

            // Calculate allowed absolute error for this fragment
            double allowed_error = std::max(fragment_masses[j], 200.0) * params.tolerance_ppm / 1e6;

            for (const auto& frag_formula : fragment_solutions) {
                double calc_mass = 0.0;
                for (int i = 0; i < FormulaAnnotation::NUM_ELEMENTS; ++i) {
                    calc_mass += frag_formula[i] * FormulaAnnotation::ATOMIC_MASSES[i];
                }
                double abs_error = std::abs(calc_mass - fragment_masses[j]);
                // Only accept if within allowed absolute error
                if (abs_error > allowed_error) continue;
                decomp.fragment_masses[j].push_back(calc_mass);
                decomp.fragment_errors_ppm[j].push_back(abs_error);
            }
            decomp.fragments[j] = std::move(fragment_solutions);
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
    fragment_params.min_dbe = -100.0;  // relaxed DBE for fragments
    fragment_params.max_dbe = 100.0;
    
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