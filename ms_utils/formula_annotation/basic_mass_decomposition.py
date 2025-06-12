import pulp
import math
from typing import List, Dict

def solve_mass_decomposition(
    target_mass: float,
    error_ppm: float,
    element_details: Dict[str, Dict[str, float]],
    rules: list = None,
    max_solutions: int = 100,
    solver_options: dict = None,
):
    """
    Performs mass decomposition to find possible elemental formulas for a given mass.

    Args:
        target_mass: The measured mass.
        error_ppm: The error tolerance in parts per million (ppm).
        element_details: A dictionary where keys are element symbols (e.g., 'C', 'H')
                         and values are dictionaries with 'mass' (float) and optionally
                         'min' (int, default 0) and 'max' (int, optional) counts.
                         Example: {'C': {'mass': 12.000000, 'min': 0}, 'H': {'mass': 1.007825}}
        rules: A list of functions. Each function takes a dictionary of PuLP variables
               (keys are element symbols) and returns a PuLP constraint expression
               (e.g., vars_dict['H'] <= 2 * vars_dict['C'] + 3) or None.
        max_solutions: Maximum number of unique solutions to find.
        solver_options: Dictionary of options for the PuLP solver (e.g., {'msg': False}).

    Returns:
        A list of dictionaries, where each dictionary represents a valid formula
        (e.g., [{'C': 6, 'H': 5, 'O': 1}, ...]).
    """
    if rules is None:
        rules = []
    if solver_options is None:
        solver_options = {"msg": False}  # Suppress solver messages by default

    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute

    print(f"Target mass: {target_mass:.6f}, Error: {error_ppm} ppm")
    print(f"Searching in mass range: [{min_mass:.6f} - {max_mass:.6f}]")

    found_formulas = []  # Stores unique formula dictionaries
    _found_formulas_canonical_set = set() # Stores canonical forms for quick uniqueness checks
    element_symbols = list(element_details.keys())

    # Initialize problem and variables ONCE
    problem = pulp.LpProblem("MassDecomposition_Persistent", pulp.LpMinimize) # Or LpProblem("...", sense=None) if no objective
    vars_dict = {}
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        if element_data["mass"] <= 0:
            raise ValueError(
                f"Element {element_symbol} has non-positive mass: {element_data['mass']}"
            )
        min_count = element_data.get("min", 0)
        # Calculate a reasonable upper bound if not provided.
        # max_mass / element_data["mass"] can be very large if element_data["mass"] is small.
        # Consider a global max atoms or per-element max if performance is an issue.
        default_ub = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        # Ensure default_ub is not excessively large, perhaps cap it.
        # For example, default_ub = min(default_ub, 1000) # Arbitrary cap
        upper_bound = element_data.get("max", default_ub)
        if upper_bound < min_count: # If user max is less than min, this element cannot exist or problem is ill-defined.
             # This could be an error or simply mean this element var should be fixed to 0 if min_count is 0.
             # For now, let PuLP handle it, or add specific logic.
             pass


        vars_dict[element_symbol] = pulp.LpVariable(
            f"{element_symbol}", # Simpler name as problem is persistent
            lowBound=min_count,
            upBound=upper_bound,
            cat=pulp.LpInteger
        )

    # Add dummy objective ONCE
    problem += 0, "DummyObjective"

    # Add mass constraints ONCE
    current_mass_expr = pulp.lpSum(
        vars_dict[el_sym] * element_details[el_sym]["mass"]
        for el_sym in element_symbols
    )
    problem += current_mass_expr >= min_mass, "MinMassConstraint"
    problem += current_mass_expr <= max_mass, "MaxMassConstraint"

    # Add heuristic rules ONCE
    vars_for_rules = vars_dict
    for i, rule_func in enumerate(rules):
        try:
            constraint = rule_func(vars_for_rules)
            if constraint is not None:
                problem += constraint, f"HeuristicRule_{i}"
        except KeyError as e:
            print(
                f"Warning: Element {e} in a rule not found in element_details. Skipping rule {i}."
            )
        except Exception as e:
            print(f"Warning: Error applying rule {i}: {e}. Skipping rule.")


    # Safety break for total solver attempts
    max_solver_attempts = max_solutions * 5 # Allow more attempts, can be tuned
    current_solver_attempts = 0

    while len(found_formulas) < max_solutions and current_solver_attempts < max_solver_attempts:
        current_solver_attempts += 1

        # Solve the problem
        # solver = pulp.PULP_CBC_CMD(**solver_options)
        # use gurobi if available, else fallback to default solver
        try:
            solver = pulp.GUROBI(**solver_options) 
        except Exception as e:
            print(f"Error initializing solver: {e}. Falling back to default solver.")
            solver = pulp.PULP_CBC_CMD(**solver_options)

        status = problem.solve(solver)

        if pulp.LpStatus[status] == "Optimal":
            current_formula: Dict[str, int] = {}
            for element_key in element_symbols:
                val = pulp.value(vars_dict[element_key])
                print(f"Element {element_key}: {val} (rounded to nearest integer)")
                if val is not None:
                    rounded_val = int(round(val)) # Round to nearest integer
                    if rounded_val > 0:
                        current_formula[element_key] = rounded_val
            
            if current_formula:
                canonical_form = tuple(sorted(current_formula.items()))
                if canonical_form not in _found_formulas_canonical_set:
                    found_formulas.append(current_formula)
                    _found_formulas_canonical_set.add(canonical_form)
                    
                    # Add constraint to exclude this newly found unique solution
                    # from future searches on the SAME problem object.
                    abs_diff_sum_terms = []
                    # Use a unique index for d_plus/d_minus variables for this specific cut
                    solution_cut_index = len(found_formulas) -1 
                    for element_key in element_symbols:
                        # prev_formula_dict is current_formula here
                        val_in_prev = current_formula.get(element_key, None)
                        if val_in_prev is None:
                            # If the element was not in the current formula, skip it
                            continue
                        x_var = vars_dict[element_key] # This is the persistent LpVariable
                        # option 1- use absolute difference with d_plus/d_minus
                        # # Ensure d_plus/d_minus variable names are unique for each cut
                        # d_plus_name = f"d_plus_{element_key}_cut{solution_cut_index}"
                        # d_minus_name = f"d_minus_{element_key}_cut{solution_cut_index}"

                        # d_plus = pulp.LpVariable(name=d_plus_name, lowBound=0, cat=pulp.LpInteger)
                        # d_minus = pulp.LpVariable(name=d_minus_name, lowBound=0, cat=pulp.LpInteger)
                        
                        # problem += (x_var - val_in_prev == d_plus - d_minus,
                        #          f"AbsDiffDef_{element_key}_cut{solution_cut_index}")
                        # abs_diff_sum_terms.append(d_plus + d_minus)
                        
                        # TODO: fix option 2- use binary features, represnting whether the new value is within 0.7 from the previous value
                        # This is a more robust way to handle numerical precision issues
                        # Use a binary variable to indicate if the current value is within 0.7 of the previous value
                        binary_var_name = f"binary_exclude_{element_key}_cut{solution_cut_index}"
                        binary_var = pulp.LpVariable(name=binary_var_name, cat=pulp.LpBinary)
                        # Add a constraint that the binary variable is 1 if the current value is within 0.7 of the previous value
                        problem += (x_var - val_in_prev <= 0.7 + (1 - binary_var) * 1000,
                                 f"BinaryWithinCut_{element_key}_cut{solution_cut_index}")
                        problem += (val_in_prev - x_var <= 0.7 + binary_var * 1000,
                                 f"BinaryWithinCut_{element_key}_cut{solution_cut_index}_reverse")
                        # Add the binary variable to the sum of terms for the exclusion cut
                        abs_diff_sum_terms.append(binary_var)

                    
                    if abs_diff_sum_terms: # Ensure there's something to sum
                        problem += (pulp.lpSum(abs_diff_sum_terms) >= 1,
                                 f"ExcludeSolCut_{solution_cut_index}")
                    else:
                        # This case should ideally not happen if element_symbols is not empty
                        print(f"Warning: No terms for exclusion cut for solution {current_formula}")

                else:
                    # This means the solver found a solution that, after rounding,
                    # is identical to one already found, despite previous exclusion cuts.
                    # This indicates potential numerical issues or that the solver is "stuck".
                    print(f"Solver returned a non-unique formula: {current_formula} (Attempt {current_solver_attempts}). This might indicate numerical precision limits.")
                    # Optionally, if this happens too often, you might want to break or implement a more robust stopping criterion.
            else:
                # Solver returned an empty formula, but was optimal.
                print(f"Warning: Solver returned an empty formula as optimal for attempt {current_solver_attempts}.")
                # This might be a valid solution if min_mass is near zero and all element counts can be zero.
                # If it's unexpected, it might indicate an issue.
                # If an empty formula is genuinely possible and undesirable, add constraints to prevent it.
                # For now, we treat it as a solution to be potentially excluded if it's the first one.
                # However, the `if current_formula:` check above already filters it from being added.
                # If an empty formula is found, and it's the *only* thing the solver can find,
                # the loop might break if no other solutions are found.
                pass # Handled by `if current_formula:`
        else:
            print(f"Solver status: {pulp.LpStatus[status]}. No more unique solutions can be found satisfying all constraints.")
            break # Exit the while loop, as no more (or no further unique) solutions

    if current_solver_attempts >= max_solver_attempts and len(found_formulas) < max_solutions:
        print(f"Warning: Reached maximum solver attempts ({max_solver_attempts}) before finding {max_solutions} unique solutions. Found {len(found_formulas)}.")

    return found_formulas



if __name__ == "__main__":
    # --- Example Usage ---
    print("Running Mass Decomposition Example...")

    # Define elements and their properties
    # Common elements for organic molecules
    elements_data = {
        "C": {"mass": 12.000000, "min": 0, "max": 50},  # Carbon
        "H": {"mass": 1.007825, "min": 0, "max": 200},  # Hydrogen
        "O": {"mass": 15.994914, "min": 0, "max": 20},  # Oxygen
        "N": {"mass": 14.003074, "min": 0, "max": 20},  # Nitrogen
        "S": {"mass": 31.972071, "min": 0, "max": 10},  # Sulfur
        "P": {"mass": 30.973762, "min": 0, "max": 5},  # Phosphorus
        'F': {"mass": 18.998403, "min": 0, "max": 5},  # Fluorine
        'Cl': {"mass": 34.968853, "min": 0, "max": 5},  # Chlorine
        'Br': {"mass": 78.918337, "min": 0, "max": 5},  # Bromine
        'I': {"mass": 126.904473, "min": 0, "max": 5},  # Iodine
        'Si': {"mass": 28.085536, "min": 0, "max": 5},  # Silicon
    }

    # Define heuristic rules
    # Rule 1: Number of Hydrogens ( DBE/Nitrogen rule like constraint)
    def rule_hydrogen_count(vars_d):
        # H <= 2*C + N + 2*S + P + 3 (example, adjust as needed)
        # A simpler common one: H <= 2*C + N + 2 (if only C,H,N,O)
        # User requested: H <= 2*C + 3
        if "H" in vars_d and "C" in vars_d:
            return vars_d["H"] <= 2 * vars_d["C"] + 3
        return None  # Rule not applicable if C or H not defined

    # Rule 2: Senior's theorem (sum of valencies must be even, or total atoms even if all monovalent/divalent etc.)
    # This is more complex to implement generally. For simplicity, let's add a ratio rule.
    def rule_oc_ratio(vars_d):
        # Example: Oxygen count should not exceed Carbon count
        if "O" in vars_d and "C" in vars_d:
            return vars_d["O"] <= vars_d["C"]
        return None

    custom_rules = [rule_hydrogen_count, rule_oc_ratio]

    # --- Test Case 3: User specified mass 123.4567 ---
    measured_mass_user = 285.136493
    error_user = 2  # ppm
    elements_data_generic = {
        "C": {"mass": 12.000000, "min": 0, "max": 100},
        "H": {"mass": 1.007825, "min": 0, "max": 200},  # 2*10+3
        "O": {"mass": 15.994914, "min": 0, "max": 20},
        "N": {"mass": 14.003074, "min": 0, "max": 20},
    }
    # user_rules = [
    #     lambda vd: vd['H'] <= 2 * vd['C'] + vd.get('N',0) + 3 if 'H' in vd and 'C' in vd else None, # H <= 2C + N + 3
    # ]

    # rules :
    # 
    print(f"\n--- Decomposing mass {measured_mass_user} +/- {error_user} ppm ---")
    solutions_user = solve_mass_decomposition(
        measured_mass_user,
        error_user,
        elements_data_generic,
        rules=custom_rules,
        max_solutions=3,
    )
    if solutions_user:
        print(f"Found {len(solutions_user)} possible formulas:")
        for i, formula in enumerate(solutions_user):
            formula_str = "".join(
                [f"{el}{count}" for el, count in sorted(formula.items())]
            )
            mass = sum(
                elements_data_generic[el]["mass"] * count
                for el, count in formula.items()
            )
            delta_ppm = ((mass - measured_mass_user) / measured_mass_user) * 1e6
            print(
                f"  {i + 1}. {formula_str} (Mass: {mass:.6f}, Delta: {delta_ppm:.2f} ppm)"
            )
    else:
        print(f"No solutions found for mass {measured_mass_user}.")
