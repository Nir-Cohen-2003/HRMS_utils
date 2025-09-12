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
                double error = calc_mass - target_mass;
                decomp.fragment_masses[j].push_back(calc_mass);
                decomp.fragment_errors_ppm[j].push_back(error);
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
            const double error = calc_mass - target;
            const double ppm = error * 1e6 / denom_report; // report relative to actual mass
            errors_ppm.push_back(ppm);
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

MassDecomposer::CleanedAndNormalizedSpectrumResult MassDecomposer::clean_and_normalize_spectrum_known_precursor(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const std::vector<double>& fragment_intensities,
    const DecompositionParams& params) {



    const size_t n = fragment_masses.size();
    MassDecomposer::CleanedAndNormalizedSpectrumResult out;
    out.masses_normalized.reserve(n);
    out.intensities.reserve(n);
    out.fragment_formulas.reserve(n);
    out.fragment_errors_ppm.reserve(n);

    // Compute all candidate formulas per fragment under the precursor constraint
    const auto fragment_solutions = decompose_spectrum_known_precursor(
        precursor_formula, fragment_masses, params);

    // Selection bookkeeping
    std::vector<bool> keep(n, false);
    std::vector<Formula> chosen_formula(n);
    std::vector<double> chosen_error_unscaled(n, 0.0); // calc_mass - target_mass

    // Helper: compute molecular mass for a formula
    auto compute_mass_for = [](const Formula& f) {
        double m = 0.0;
        for (int k = 0; k < FormulaAnnotation::NUM_ELEMENTS; ++k) {
            m += f[k] * FormulaAnnotation::ATOMIC_MASSES[k];
        }
        return m;
    };

    // 1) Initial weighted linear fit err ~ a + b * mass using single-option fragments only.
    //    Weight = fragment mass. Rationale: higher mass fragments tend to have lower relative ppm.
    double Sw = 0.0, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const auto& formulas = fragment_solutions[i];
        if (formulas.size() == 1) {
            const double target = fragment_masses[i];
            const double calc_mass = compute_mass_for(formulas[0]);
            const double err = calc_mass - target;
            const double w = target; // mass-weight

            chosen_formula[i] = formulas[0];
            chosen_error_unscaled[i] = err;
            keep[i] = true;

            if (w > 0.0) {
                Sw  += w;
                Sx  += w * target;
                Sy  += w * err;
                Sxx += w * target * target;
                Sxy += w * target * err;
            }
        }
    }

    auto finalize_fit = [](double Sw, double Sx, double Sy, double Sxx, double Sxy) {
        double a = 0.0, b = 0.0;
        const double denom = Sw * Sxx - Sx * Sx;
        if (Sw > 0.0 && std::abs(denom) > 1e-12) {
            b = (Sw * Sxy - Sx * Sy) / denom;
            a = (Sy - b * Sx) / Sw;
        } else if (Sw > 0.0) {
            // Fallback to additive-only correction if slope is ill-conditioned
            b = 0.0;
            a = Sy / Sw;
        } else {
            a = 0.0;
            b = 0.0;
        }
        return std::pair<double,double>(a, b);
    };

    auto [a, b] = finalize_fit(Sw, Sx, Sy, Sxx, Sxy);

    // 2) For multi-option fragments, pick the candidate whose error is closest to the current model:
    //    minimize |(calc - target) - (a + b * target)|.
    for (size_t i = 0; i < n; ++i) {
        const auto& formulas = fragment_solutions[i];
        if (formulas.empty() || keep[i]) continue;

        const double target = fragment_masses[i];
        const double model = a + b * target;

        double best_abs = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        double best_err = 0.0;

        for (int c = 0; c < static_cast<int>(formulas.size()); ++c) {
            const double calc_mass = compute_mass_for(formulas[c]);
            const double err = calc_mass - target;
            const double dev = std::abs(err - model); // abs vs squared yields same argmin
            if (dev < best_abs) {
                best_abs = dev;
                best_idx = c;
                best_err = err;
            }
        }

        if (best_idx >= 0) {
            chosen_formula[i] = formulas[best_idx];
            chosen_error_unscaled[i] = best_err;
            keep[i] = true;
        }
    }

    // 3) Refit a and b using all selected fragments (single + chosen multi) with mass weights.
    Sw = Sx = Sy = Sxx = Sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        const double target = fragment_masses[i];
        const double err = chosen_error_unscaled[i];
        const double w = target; // mass-weight
        if (w > 0.0) {
            Sw  += w;
            Sx  += w * target;
            Sy  += w * err;
            Sxx += w * target * target;
            Sxy += w * target * err;
        }
    }
    std::tie(a, b) = finalize_fit(Sw, Sx, Sy, Sxx, Sxy);

    // 4) Emit kept fragments in original order with normalized masses and final error reporting.
    //    Normalized mass is target + (a + b*target). Error after normalization = err - (a + b*target).
    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;

        const double target = fragment_masses[i];
        const double correction = a + b * target; // additive + multiplicative (via mass term)
        const double normalized_mass = target + correction;

        const double err_after_norm = chosen_error_unscaled[i] - correction; // calc - normalized
        const double denom_report = (normalized_mass != 0.0)
            ? normalized_mass
            : std::max(target, 200.0);
        const double ppm_after_norm = (err_after_norm * 1e6) / denom_report;

        out.masses_normalized.push_back(normalized_mass);
        out.intensities.push_back(fragment_intensities[i]);
        out.fragment_formulas.push_back(chosen_formula[i]);
        out.fragment_errors_ppm.push_back(ppm_after_norm);
    }

    return out;
}

std::vector<MassDecomposer::CleanedAndNormalizedSpectrumResult>
MassDecomposer::clean_and_normalize_spectra_known_precursor_parallel(
    const std::vector<MassDecomposer::CleanSpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {

    const int n = static_cast<int>(spectra.size());
    std::vector<MassDecomposer::CleanedAndNormalizedSpectrumResult> all_results(n);

    #pragma omp parallel
    {
        // Thread-local decomposer instance to call non-static member
        MassDecomposer thread_decomposer(params.min_bounds, params.max_bounds);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            const auto& s = spectra[i];
            all_results[i] = thread_decomposer.clean_and_normalize_spectrum_known_precursor(
                s.precursor_formula, s.fragment_masses, s.fragment_intensities, params);
        }
    }
    return all_results;
}