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

std::vector<std::vector<FormulaWithString> > MassDecomposer::decompose_parallel_verbose(
    const std::vector<double>& target_masses,
    const DecompositionParams& params) {
    std::vector<std::vector<Formula> > base_results = decompose_parallel(target_masses, params);
    std::vector<std::vector<FormulaWithString> > verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const std::vector<Formula>& formulas = base_results[i];
        std::vector<FormulaWithString>& verbose_vec = verbose_results[i];
        verbose_vec.reserve(formulas.size());
        for (std::size_t j = 0; j < formulas.size(); ++j) {
            const Formula& formula = formulas[j];
            FormulaWithString entry;
            entry.formula = formula;
            entry.formula_string = formula_to_string(formula);
            verbose_vec.push_back(entry);
        }
    }
    return verbose_results;
}

std::vector<std::vector<FormulaWithString> > MassDecomposer::decompose_masses_parallel_per_bounds_verbose(
    const std::vector<double>& target_masses,
    const std::vector<std::pair<Formula, Formula> >& per_mass_bounds,
    const DecompositionParams& params) {
    std::vector<std::vector<Formula> > base_results = decompose_masses_parallel_per_bounds(target_masses, per_mass_bounds, params);
    std::vector<std::vector<FormulaWithString> > verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const std::vector<Formula>& formulas = base_results[i];
        std::vector<FormulaWithString>& verbose_vec = verbose_results[i];
        verbose_vec.reserve(formulas.size());
        for (std::size_t j = 0; j < formulas.size(); ++j) {
            const Formula& formula = formulas[j];
            FormulaWithString entry;
            entry.formula = formula;
            entry.formula_string = formula_to_string(formula);
            verbose_vec.push_back(entry);
        }
    }
    return verbose_results;
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

ProperSpectrumResultsVerbose MassDecomposer::decompose_spectrum_verbose(
    double precursor_mass,
    const std::vector<double>& fragment_masses,
    const DecompositionParams& params) {
    ProperSpectrumResults base = decompose_spectrum(precursor_mass, fragment_masses, params);
    ProperSpectrumResultsVerbose verbose;
    verbose.decompositions.reserve(base.decompositions.size());
    for (std::size_t i = 0; i < base.decompositions.size(); ++i) {
        const SpectrumDecomposition& src = base.decompositions[i];
        SpectrumDecompositionVerbose dst;
        dst.precursor = src.precursor;
        dst.precursor_string = formula_to_string(src.precursor);
        dst.precursor_mass = src.precursor_mass;
        dst.precursor_error_ppm = src.precursor_error_ppm;
        dst.fragments = src.fragments;
        dst.fragment_masses = src.fragment_masses;
        dst.fragment_errors_ppm = src.fragment_errors_ppm;
        dst.fragment_strings.resize(src.fragments.size());
        for (std::size_t j = 0; j < src.fragments.size(); ++j) {
            const std::vector<Formula>& fragment_formulas = src.fragments[j];
            std::vector<std::string>& fragment_strings = dst.fragment_strings[j];
            fragment_strings.reserve(fragment_formulas.size());
            for (std::size_t k = 0; k < fragment_formulas.size(); ++k) {
                fragment_strings.push_back(formula_to_string(fragment_formulas[k]));
            }
        }
        verbose.decompositions.push_back(dst);
    }
    return verbose;
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

std::vector<ProperSpectrumResultsVerbose> MassDecomposer::decompose_spectra_parallel_verbose(
    const std::vector<Spectrum>& spectra,
    const DecompositionParams& params) {
    std::vector<ProperSpectrumResults> base_results = decompose_spectra_parallel(spectra, params);
    std::vector<ProperSpectrumResultsVerbose> verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const ProperSpectrumResults& base = base_results[i];
        ProperSpectrumResultsVerbose& dst = verbose_results[i];
        dst.decompositions.reserve(base.decompositions.size());
        for (std::size_t j = 0; j < base.decompositions.size(); ++j) {
            const SpectrumDecomposition& src = base.decompositions[j];
            SpectrumDecompositionVerbose verbose_decomp;
            verbose_decomp.precursor = src.precursor;
            verbose_decomp.precursor_string = formula_to_string(src.precursor);
            verbose_decomp.precursor_mass = src.precursor_mass;
            verbose_decomp.precursor_error_ppm = src.precursor_error_ppm;
            verbose_decomp.fragments = src.fragments;
            verbose_decomp.fragment_masses = src.fragment_masses;
            verbose_decomp.fragment_errors_ppm = src.fragment_errors_ppm;
            verbose_decomp.fragment_strings.resize(src.fragments.size());
            for (std::size_t k = 0; k < src.fragments.size(); ++k) {
                const std::vector<Formula>& fragment_formulas = src.fragments[k];
                std::vector<std::string>& fragment_strings = verbose_decomp.fragment_strings[k];
                fragment_strings.reserve(fragment_formulas.size());
                for (std::size_t m = 0; m < fragment_formulas.size(); ++m) {
                    fragment_strings.push_back(formula_to_string(fragment_formulas[m]));
                }
            }
            dst.decompositions.push_back(verbose_decomp);
        }
    }
    return verbose_results;
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

std::vector<ProperSpectrumResultsVerbose> MassDecomposer::decompose_spectra_parallel_per_bounds_verbose(
    const std::vector<SpectrumWithBounds>& spectra,
    const DecompositionParams& params) {
    std::vector<ProperSpectrumResults> base_results = decompose_spectra_parallel_per_bounds(spectra, params);
    std::vector<ProperSpectrumResultsVerbose> verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const ProperSpectrumResults& base = base_results[i];
        ProperSpectrumResultsVerbose& dst = verbose_results[i];
        dst.decompositions.reserve(base.decompositions.size());
        for (std::size_t j = 0; j < base.decompositions.size(); ++j) {
            const SpectrumDecomposition& src = base.decompositions[j];
            SpectrumDecompositionVerbose verbose_decomp;
            verbose_decomp.precursor = src.precursor;
            verbose_decomp.precursor_string = formula_to_string(src.precursor);
            verbose_decomp.precursor_mass = src.precursor_mass;
            verbose_decomp.precursor_error_ppm = src.precursor_error_ppm;
            verbose_decomp.fragments = src.fragments;
            verbose_decomp.fragment_masses = src.fragment_masses;
            verbose_decomp.fragment_errors_ppm = src.fragment_errors_ppm;
            verbose_decomp.fragment_strings.resize(src.fragments.size());
            for (std::size_t k = 0; k < src.fragments.size(); ++k) {
                const std::vector<Formula>& fragment_formulas = src.fragments[k];
                std::vector<std::string>& fragment_strings = verbose_decomp.fragment_strings[k];
                fragment_strings.reserve(fragment_formulas.size());
                for (std::size_t m = 0; m < fragment_formulas.size(); ++m) {
                    fragment_strings.push_back(formula_to_string(fragment_formulas[m]));
                }
            }
            dst.decompositions.push_back(verbose_decomp);
        }
    }
    return verbose_results;
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

std::vector<std::vector<FormulaWithString> > MassDecomposer::decompose_spectrum_known_precursor_verbose(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const DecompositionParams& params) {
    std::vector<std::vector<Formula> > base_results = decompose_spectrum_known_precursor(precursor_formula, fragment_masses, params);
    std::vector<std::vector<FormulaWithString> > verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const std::vector<Formula>& formulas = base_results[i];
        std::vector<FormulaWithString>& verbose_vec = verbose_results[i];
        verbose_vec.reserve(formulas.size());
        for (std::size_t j = 0; j < formulas.size(); ++j) {
            const Formula& formula = formulas[j];
            FormulaWithString entry;
            entry.formula = formula;
            entry.formula_string = formula_to_string(formula);
            verbose_vec.push_back(entry);
        }
    }
    return verbose_results;
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

std::vector<std::vector<std::vector<FormulaWithString> > > MassDecomposer::decompose_spectra_known_precursor_parallel_verbose(
    const std::vector<SpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {
    std::vector<std::vector<std::vector<Formula> > > base_results = decompose_spectra_known_precursor_parallel(spectra, params);
    std::vector<std::vector<std::vector<FormulaWithString> > > verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const std::vector<std::vector<Formula> >& spectrum_formulas = base_results[i];
        std::vector<std::vector<FormulaWithString> >& verbose_spectrum = verbose_results[i];
        verbose_spectrum.resize(spectrum_formulas.size());
        for (std::size_t j = 0; j < spectrum_formulas.size(); ++j) {
            const std::vector<Formula>& fragment_formulas = spectrum_formulas[j];
            std::vector<FormulaWithString>& verbose_fragment = verbose_spectrum[j];
            verbose_fragment.reserve(fragment_formulas.size());
            for (std::size_t k = 0; k < fragment_formulas.size(); ++k) {
                const Formula& formula = fragment_formulas[k];
                FormulaWithString entry;
                entry.formula = formula;
                entry.formula_string = formula_to_string(formula);
                verbose_fragment.push_back(entry);
            }
        }
    }
    return verbose_results;
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

MassDecomposer::CleanedSpectrumResultVerbose MassDecomposer::clean_spectrum_known_precursor_verbose(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const std::vector<double>& fragment_intensities,
    const DecompositionParams& params) {
    CleanedSpectrumResult base = clean_spectrum_known_precursor(
        precursor_formula,
        fragment_masses,
        fragment_intensities,
        params
    );
    CleanedSpectrumResultVerbose verbose;
    verbose.masses = base.masses;
    verbose.intensities = base.intensities;
    verbose.fragment_formulas = base.fragment_formulas;
    verbose.fragment_errors_ppm = base.fragment_errors_ppm;
    verbose.fragment_formulas_strings.resize(base.fragment_formulas.size());
    for (std::size_t i = 0; i < base.fragment_formulas.size(); ++i) {
        const std::vector<Formula>& formulas = base.fragment_formulas[i];
        std::vector<std::string>& strings = verbose.fragment_formulas_strings[i];
        strings.reserve(formulas.size());
        for (std::size_t j = 0; j < formulas.size(); ++j) {
            strings.push_back(formula_to_string(formulas[j]));
        }
    }
    return verbose;
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

std::vector<MassDecomposer::CleanedSpectrumResultVerbose>
MassDecomposer::clean_spectra_known_precursor_parallel_verbose(
    const std::vector<MassDecomposer::CleanSpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {
    std::vector<CleanedSpectrumResult> base_results = clean_spectra_known_precursor_parallel(spectra, params);
    std::vector<CleanedSpectrumResultVerbose> verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const CleanedSpectrumResult& base = base_results[i];
        CleanedSpectrumResultVerbose verbose;
        verbose.masses = base.masses;
        verbose.intensities = base.intensities;
        verbose.fragment_formulas = base.fragment_formulas;
        verbose.fragment_errors_ppm = base.fragment_errors_ppm;
        verbose.fragment_formulas_strings.resize(base.fragment_formulas.size());
        for (std::size_t j = 0; j < base.fragment_formulas.size(); ++j) {
            const std::vector<Formula>& formulas = base.fragment_formulas[j];
            std::vector<std::string>& strings = verbose.fragment_formulas_strings[j];
            strings.reserve(formulas.size());
            for (std::size_t k = 0; k < formulas.size(); ++k) {
                strings.push_back(formula_to_string(formulas[k]));
            }
        }
        verbose_results[i] = verbose;
    }
    return verbose_results;
}

MassDecomposer::CleanedAndNormalizedSpectrumResult MassDecomposer::clean_and_normalize_spectrum_known_precursor(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const std::vector<double>& fragment_intensities,
    double precursor_mass,
    double max_allowed_normalized_mass_error_ppm,
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

    // Compute precursor modeled mass and error once; will be used as an extra calibration point.
    const double precursor_calc_mass = compute_mass_for(precursor_formula);
    const double precursor_err = precursor_calc_mass - precursor_mass;

    // 1) Initial weighted linear fit err ~ a + b * mass using single-option fragments only.
    //    Weight = fragment mass. Rationale: higher mass fragments tend to have lower relative ppm.
    double Sw = 0.0, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;

    // Include the precursor as an additional calibration point in the initial fit.
    // Why: improves stability when few single-option fragments exist.
    if (precursor_mass > 0.0) {
        const double w_prec = precursor_mass;
        Sw  += w_prec;
        Sx  += w_prec * precursor_mass;
        Sy  += w_prec * precursor_err;
        Sxx += w_prec * precursor_mass * precursor_mass;
        Sxy += w_prec * precursor_mass * precursor_err;
    }

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

    // Include the precursor as a calibration point in the refit as well.
    if (precursor_mass > 0.0) {
        const double w_prec = precursor_mass;
        Sw  += w_prec;
        Sx  += w_prec * precursor_mass;
        Sy  += w_prec * precursor_err;
        Sxx += w_prec * precursor_mass * precursor_mass;
        Sxy += w_prec * precursor_mass * precursor_err;
    }

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

    // 4) Compute normalized results for all kept fragments, then filter by max_allowed_normalized_mass_error_ppm.
    std::vector<double> tmp_masses_normalized;
    std::vector<double> tmp_intensities;
    std::vector<Formula> tmp_fragment_formulas;
    std::vector<double> tmp_fragment_errors_ppm;
    std::vector<double> tmp_err_after_norm_ppm_abs; // for thresholding (ppm)

    tmp_masses_normalized.reserve(n);
    tmp_intensities.reserve(n);
    tmp_fragment_formulas.reserve(n);
    tmp_fragment_errors_ppm.reserve(n);
    tmp_err_after_norm_ppm_abs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;

        const double target = fragment_masses[i];
        const double correction = a + b * target; // additive + multiplicative (via mass term)
        const double normalized_mass = target + correction;

        const double err_after_norm = chosen_error_unscaled[i] - correction; // calc - normalized
        const double denom_report = std::max(normalized_mass, 200.0); // ppm denom
        const double ppm_after_norm = (err_after_norm * 1e6) / denom_report;

        tmp_masses_normalized.push_back(normalized_mass);
        tmp_intensities.push_back(fragment_intensities[i]);
        tmp_fragment_formulas.push_back(chosen_formula[i]);
        tmp_fragment_errors_ppm.push_back(ppm_after_norm);
        tmp_err_after_norm_ppm_abs.push_back(std::abs(ppm_after_norm));
    }

    // Apply final filtering based on absolute normalized ppm threshold.
    for (size_t i = 0; i < tmp_masses_normalized.size(); ++i) {
        if (tmp_err_after_norm_ppm_abs[i] <= max_allowed_normalized_mass_error_ppm) {
            out.masses_normalized.push_back(tmp_masses_normalized[i]);
            out.intensities.push_back(tmp_intensities[i]);
            out.fragment_formulas.push_back(tmp_fragment_formulas[i]);
            out.fragment_errors_ppm.push_back(tmp_fragment_errors_ppm[i]);
        }
    }

    return out;
}

MassDecomposer::CleanedAndNormalizedSpectrumResultVerbose
MassDecomposer::clean_and_normalize_spectrum_known_precursor_verbose(
    const Formula& precursor_formula,
    const std::vector<double>& fragment_masses,
    const std::vector<double>& fragment_intensities,
    double precursor_mass,
    double max_allowed_normalized_mass_error_ppm,
    const DecompositionParams& params) {
    CleanedAndNormalizedSpectrumResult base = clean_and_normalize_spectrum_known_precursor(
        precursor_formula,
        fragment_masses,
        fragment_intensities,
        precursor_mass,
        max_allowed_normalized_mass_error_ppm,
        params
    );
    CleanedAndNormalizedSpectrumResultVerbose verbose;
    verbose.masses_normalized = base.masses_normalized;
    verbose.intensities = base.intensities;
    verbose.fragment_formulas = base.fragment_formulas;
    verbose.fragment_errors_ppm = base.fragment_errors_ppm;
    verbose.fragment_formulas_strings.reserve(base.fragment_formulas.size());
    for (std::size_t i = 0; i < base.fragment_formulas.size(); ++i) {
        verbose.fragment_formulas_strings.push_back(formula_to_string(base.fragment_formulas[i]));
    }
    return verbose;
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
                s.precursor_formula,
                s.fragment_masses,
                s.fragment_intensities,
                s.precursor_mass,
                s.max_allowed_normalized_mass_error_ppm,
                params);
        }
    }
    return all_results;
}

std::vector<MassDecomposer::CleanedAndNormalizedSpectrumResultVerbose>
MassDecomposer::clean_and_normalize_spectra_known_precursor_parallel_verbose(
    const std::vector<MassDecomposer::CleanSpectrumWithKnownPrecursor>& spectra,
    const DecompositionParams& params) {
    std::vector<CleanedAndNormalizedSpectrumResult> base_results = clean_and_normalize_spectra_known_precursor_parallel(spectra, params);
    std::vector<CleanedAndNormalizedSpectrumResultVerbose> verbose_results(base_results.size());
    for (std::size_t i = 0; i < base_results.size(); ++i) {
        const CleanedAndNormalizedSpectrumResult& base = base_results[i];
        CleanedAndNormalizedSpectrumResultVerbose verbose;
        verbose.masses_normalized = base.masses_normalized;
        verbose.intensities = base.intensities;
        verbose.fragment_formulas = base.fragment_formulas;
        verbose.fragment_errors_ppm = base.fragment_errors_ppm;
        verbose.fragment_formulas_strings.reserve(base.fragment_formulas.size());
        for (std::size_t j = 0; j < base.fragment_formulas.size(); ++j) {
            verbose.fragment_formulas_strings.push_back(formula_to_string(base.fragment_formulas[j]));
        }
        verbose_results[i] = verbose;
    }
    return verbose_results;
}