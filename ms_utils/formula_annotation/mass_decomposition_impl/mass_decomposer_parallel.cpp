#include "mass_decomposer_common.hpp"
#include <omp.h>

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

std::vector<std::vector<Formula>> MassDecomposer::decompose_masses_parallel_per_bounds(
    const std::vector<double>& target_masses,
    const std::vector<std::vector<Element>>& per_mass_bounds,
    const DecompositionParams& params) {

    int n_masses = static_cast<int>(target_masses.size());
    std::vector<std::vector<Formula>> all_results(n_masses);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_masses; ++i) {
        // Create a decomposer instance with specific bounds for this mass
        MassDecomposer thread_decomposer(per_mass_bounds[i], params.strategy);
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
        for (const auto& pair : precursor_formula) {
            auto it = atomic_masses_.find(pair.first);
            if (it != atomic_masses_.end()) {
                decomp.precursor_mass += it->second * pair.second;
            }
        }
        decomp.precursor_error_ppm = std::abs(decomp.precursor_mass - precursor_mass) / precursor_mass * 1e6;

        // Create bounds for fragment decomposition based on precursor formula
        std::vector<Element> fragment_bounds = create_bounds_from_formula(precursor_formula);

        // Create a decomposer for the fragments with the specific bounds from the precursor
        MassDecomposer fragment_decomposer(fragment_bounds, params.strategy);
        DecompositionParams fragment_params = params;
        fragment_params.min_dbe = 0.0;  // relaxed DBE for fragments
        
        // Decompose each fragment serially for this precursor candidate
        decomp.fragments.resize(fragment_masses.size());
        decomp.fragment_masses.resize(fragment_masses.size());
        decomp.fragment_errors_ppm.resize(fragment_masses.size());

        for (size_t j = 0; j < fragment_masses.size(); ++j) {
            auto fragment_solutions = fragment_decomposer.decompose(fragment_masses[j], fragment_params);
            
            // Compute masses and errors for the found fragment formulas
            for (const auto& frag_formula : fragment_solutions) {
                double calc_mass = 0.0;
                for (const auto& pr : frag_formula) {
                    auto it = atomic_masses_.find(pr.first);
                    if (it != atomic_masses_.end()) calc_mass += it->second * pr.second;
                }
                double error_ppm = std::abs(calc_mass - fragment_masses[j]) / fragment_masses[j] * 1e6;
                decomp.fragment_masses[j].push_back(calc_mass);
                decomp.fragment_errors_ppm[j].push_back(error_ppm);
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

std::vector<ProperSpectrumResults> MassDecomposer::decompose_spectra_parallel_per_bounds(
    const std::vector<SpectrumWithBounds>& spectra,
    const DecompositionParams& params) {
    
    int n_spectra = static_cast<int>(spectra.size());
    std::vector<ProperSpectrumResults> all_results(n_spectra);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_spectra; ++i) {
        const auto& spectrum = spectra[i];
        // Each thread creates a decomposer with the specific bounds for the current spectrum
        MassDecomposer thread_decomposer(spectrum.precursor_bounds, params.strategy);
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
    
    // Create bounds for fragment decomposition based on precursor formula
    std::vector<Element> fragment_bounds = create_bounds_from_formula(precursor_formula);
    
    // Create a decomposer for the fragments with the specific bounds from the precursor
    MassDecomposer fragment_decomposer(fragment_bounds, params.strategy);
    DecompositionParams fragment_params = params;
    fragment_params.min_dbe = 0.0;  // relaxed DBE for fragments
    
    // Decompose each fragment
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
    
    // Use OpenMP for true parallel processing of spectra
    #pragma omp parallel
    {
        // Each thread needs its own decomposer instance to avoid race conditions
        MassDecomposer thread_decomposer(elements_, params.strategy);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_spectra; ++i) {
            const SpectrumWithKnownPrecursor& spectrum = spectra[i];
            all_results[i] = thread_decomposer.decompose_spectrum_known_precursor(
                spectrum.precursor_formula, spectrum.fragment_masses, params);
        }
    }
    
    return all_results;
}