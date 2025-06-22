import pulp
import math
from typing import List, Dict
from time import perf_counter
from functools import wraps

def profile_function(func):
    """Decorator to profile individual functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"PROFILE: {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def solve_mass_decomposition(
    target_mass: float,
    error_ppm: float,
    element_details: Dict[str, Dict[str, float]],
    rules: list = None,
    max_solutions: int = 100,
    solver_options: dict = None,
    enable_profiling: bool = False,
):
    """
    One-hot encoding version of mass decomposition using binary variables
    for each possible count of each element.
    """
    if rules is None:
        rules = []
    if solver_options is None:
        solver_options = {"msg": False}

    # Start total timing
    total_start = perf_counter()
    
    # Timing dictionary to track all phases
    timing_profile = {
        "setup": 0.0,
        "variable_creation": 0.0,
        "constraint_creation": 0.0,
        "rule_application": 0.0,
        "solving": 0.0,
        "solution_extraction": 0.0,
        "solution_exclusion": 0.0,
        "total": 0.0
    }

    # Setup phase
    setup_start = perf_counter()
    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute

    print(f"Target mass: {target_mass:.6f}, Error: {error_ppm} ppm")
    print(f"Searching in mass range: [{min_mass:.6f} - {max_mass:.6f}]")

    found_formulas = []
    element_symbols = list(element_details.keys())
    timing_profile["setup"] = perf_counter() - setup_start

    # Variable creation phase
    var_creation_start = perf_counter()
    
    # Create problem
    prob = pulp.LpProblem("MassDecomposition_OneHot")

    # Create one-hot binary variables for each element and possible count
    choices = {}  # choices[element][count] = binary variable
    count_ranges = {}  # Store valid count ranges for each element
    
    total_variables = 0
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        # max_count = min(element_data.get("max", default_max), 100)  # Cap at 100 for performance
        max_count = element_data.get("max", max_allowed_count)
        max_count = min(max_count, max_allowed_count)  # make sure we won't consider counts that would exceed max_mass
        count_ranges[element_symbol] = range(min_count, max_count + 1)
        choices[element_symbol] = {}
        
        for count in count_ranges[element_symbol]:
            choices[element_symbol][count] = pulp.LpVariable(
                f"Choice_{element_symbol}_{count}", cat="Binary"
            )
            total_variables += 1
    
    print(f"PROFILE: Created {total_variables} binary variables")
    timing_profile["variable_creation"] = perf_counter() - var_creation_start

    # Constraint creation phase
    constraint_start = perf_counter()

    # Constraint: exactly one count must be chosen for each element
    for element_symbol in element_symbols:
        prob += pulp.lpSum([
            choices[element_symbol][count] 
            for count in count_ranges[element_symbol]
        ]) == 1, f"OneCount_{element_symbol}"

    # Create auxiliary variables for actual element counts (for rules only)
    element_counts = {}
    for element_symbol in element_symbols:
        element_counts[element_symbol] = pulp.lpSum([
            choices[element_symbol][count] * count
            for count in count_ranges[element_symbol]
        ])

    # Mass constraint - create directly without intermediate auxiliary variables
    mass_contributions = []
    for element_symbol in element_symbols:
        for count in count_ranges[element_symbol]:
            mass_contributions.append(
                choices[element_symbol][count] * count * element_details[element_symbol]["mass"]
            )
    
    total_mass = pulp.lpSum(mass_contributions)
    
    # Add mass bounds as constraints
    prob += total_mass >= min_mass, "MinMassConstraint"
    prob += total_mass <= max_mass, "MaxMassConstraint"
    
    
    timing_profile["constraint_creation"] = perf_counter() - constraint_start
    print(f"DEBUG: Mass constraints set to [{min_mass:.6f}, {max_mass:.6f}]")

    # Rule application phase
    rule_start = perf_counter()
    
    # Add heuristic rules if provided
    if rules:
        # Apply rules using the one-hot choice variables directly
        for i, rule_func in enumerate(rules):
            try:
                # Check if rule function expects one-hot parameters
                import inspect
                sig = inspect.signature(rule_func)
                if len(sig.parameters) == 2:
                    # New one-hot rule format
                    constraint = rule_func(choices, count_ranges)
                else:
                    # Legacy rule format using auxiliary variables
                    constraint = rule_func(element_counts)
                
                if constraint is not None:
                    prob += constraint, f"HeuristicRule_{i}"
                    print(f"DEBUG: Added rule {i}")
            except Exception as e:
                print(f"Warning: Error applying rule {i}: {e}. Skipping rule.")
    
    timing_profile["rule_application"] = perf_counter() - rule_start

    # Open output file for writing solutions
    solution_count = 0
    total_solving_time = 0.0
    total_extraction_time = 0.0
    total_exclusion_time = 0.0
    
    solver_setup_start = perf_counter()
    try:
        solver = pulp.GUROBI(
            msg=False,
            IntFeasTol=1e-9,  # Integer tolerance
        )
        # print("Using GUROBI solver...")
    except Exception:
        solver = pulp.PULP_CBC_CMD(**solver_options)
        # print("Using default solver...")
    timing_profile["solver_setup"] = perf_counter() - solver_setup_start
    while solution_count < max_solutions:
        # Solve the problem
        solve_start = perf_counter()
        status = prob.solve(solver)
        solve_time = perf_counter() - solve_start
        total_solving_time += solve_time
        
        if enable_profiling:
            print(f"PROFILE: Solve iteration {solution_count + 1} took {solve_time:.4f} seconds")
        
        if pulp.LpStatus[status] == "Optimal":
            # Extract the solution and verify mass constraint
            extraction_start = perf_counter()
            
            current_formula = {}
            actual_mass = 0.0
            
            for element_symbol in element_symbols:
                element_count = 0
                choices_found = 0
                for count in count_ranges[element_symbol]:
                    choice_value = pulp.value(choices[element_symbol][count])
                    if abs(choice_value - 1.0) < 1e-6:  # More robust floating point comparison
                        element_count = count
                        choices_found += 1
                
                # Verify one-hot constraint
                if choices_found != 1:
                    print(f"ERROR: Element {element_symbol} has {choices_found} choices set to 1!")
                    # Debug: show all choice values for this element
                    for count in count_ranges[element_symbol]:
                        val = pulp.value(choices[element_symbol][count])
                        if val > 1e-6:
                            print(f"  {element_symbol}[{count}] = {val}")
                
                if element_count > 0:
                    current_formula[element_symbol] = element_count
                actual_mass += element_count * element_details[element_symbol]["mass"]
            
            # Double-check mass constraint
            delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
            
            extraction_time = perf_counter() - extraction_start
            total_extraction_time += extraction_time
            
            if actual_mass < min_mass or actual_mass > max_mass:
                print(f"ERROR: Solution violates mass constraint!")
                print(f"  Formula: {current_formula}")
                print(f"  Actual mass: {actual_mass:.6f}")
                print(f"  Target mass: {target_mass:.6f}")
                print(f"  Allowed range: [{min_mass:.6f}, {max_mass:.6f}]")
                print(f"  Delta: {delta_ppm:.2f} ppm")
                print(f"  Error tolerance: {error_ppm} ppm")
                
                # Check what the solver thinks the mass is
                solver_mass = pulp.value(total_mass)
                print(f"  Solver calculated mass: {solver_mass:.6f}")
                break
            print(f"solution {solution_count + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
            found_formulas.append(current_formula)
            solution_count += 1
            
            # Add constraint to exclude this exact solution (Sudoku-style)
            exclusion_start = perf_counter()
            
            current_solution_vars = []
            for element_symbol in element_symbols:
                for count in count_ranges[element_symbol]:
                    if pulp.value(choices[element_symbol][count]) == 1:
                        current_solution_vars.append(choices[element_symbol][count])
            
            # Constraint: at least one of these variables must be 0 in future solutions
            if current_solution_vars:
                prob += (
                    pulp.lpSum(current_solution_vars) <= len(current_solution_vars) - 1,
                    f"ExcludeSolution_{solution_count}"
                )
            
            exclusion_time = perf_counter() - exclusion_start
            total_exclusion_time += exclusion_time
            
        else:
            print(f"Solver status: {pulp.LpStatus[status]}. No more solutions found.")
            break

    # Final timing calculations
    timing_profile["solving"] = total_solving_time
    timing_profile["solution_extraction"] = total_extraction_time
    timing_profile["solution_exclusion"] = total_exclusion_time
    timing_profile["total"] = perf_counter() - total_start

    # Print detailed timing profile
    print("\n" + "="*60)
    print("DETAILED TIMING PROFILE")
    print("="*60)
    for phase, time_taken in timing_profile.items():
        percentage = (time_taken / timing_profile["total"]) * 100 if timing_profile["total"] > 0 else 0
        print(f"{phase.replace('_', ' ').title():<25}: {time_taken:.4f}s ({percentage:.1f}%)")
    
    print(f"\nSolutions found: {len(found_formulas)}")
    print(f"Average solve time per iteration: {total_solving_time/max(solution_count, 1):.4f}s")
    if solution_count > 0:
        print(f"Total iterations: {solution_count}")
        print(f"Time per solution: {timing_profile['total']/solution_count:.4f}s")

    return found_formulas


if __name__ == "__main__":
    # --- Example Usage ---
    print("Running Mass Decomposition Example with One-Hot Encoding...")

    # Define elements and their properties
    elements_data_generic = {
        "C": {"mass": 12.000000, "min": 15, "max": 19},
        "H": {"mass": 1.007825, "min": 0, "max": 50},
        "O": {"mass": 15.994914, "min": 0, "max": 20},
        "N": {"mass": 14.003074, "min": 0, "max": 10},
        "S": {"mass": 31.972071, "min": 0, "max": 0},
        "P": {"mass": 30.973762, "min": 0, "max": 5},
        "F": {"mass": 18.998403, "min": 0, "max": 20},
        "Cl": {"mass": 34.968853, "min": 0, "max": 0},
        "Br": {"mass": 78.918337, "min": 0, "max": 0},
        "I": {"mass": 126.904473, "min": 0, "max": 2},
    }

    # Define heuristic rules for one-hot encoding
    def rule_DBE_onehot(choices, count_ranges):
        """
        DBE (Double Bond Equivalent) rule: H <= 2*C + 3
        Logic: This prevents chemically unrealistic formulas with too many hydrogens
        relative to carbons. In one-hot encoding, we express this as:
        sum(h_count * choice_H[h_count]) <= 2 * sum(c_count * choice_C[c_count]) + 3
        """
        if "H" not in choices or "C" not in choices:
            return None
        
        h_sum = pulp.lpSum([
            count * choices["H"][count] 
            for count in count_ranges["H"]
        ])
        c_sum = pulp.lpSum([
            count * choices["C"][count] 
            for count in count_ranges["C"]
        ])
        
        return h_sum <= 2 * c_sum + 3

    def rule_oc_ratio_onehot(choices, count_ranges):
        """
        Oxygen-Carbon ratio rule: O <= C
        Logic: This prevents formulas with more oxygens than carbons, which are
        chemically less common in most organic compounds.
        """
        if "O" not in choices or "C" not in choices:
            return None
        
        o_sum = pulp.lpSum([
            count * choices["O"][count] 
            for count in count_ranges["O"]
        ])
        c_sum = pulp.lpSum([
            count * choices["C"][count] 
            for count in count_ranges["C"]
        ])
        
        return o_sum <= c_sum

    def rule_nitrogen_limit_onehot(choices, count_ranges):
        """
        Nitrogen limit rule: N <= C/2 + 1
        Logic: Most organic compounds have relatively few nitrogens compared to carbons.
        This rule helps eliminate unlikely high-nitrogen formulas.
        """
        if "N" not in choices or "C" not in choices:
            return None
        
        n_sum = pulp.lpSum([
            count * choices["N"][count] 
            for count in count_ranges["N"]
        ])
        c_sum = pulp.lpSum([
            count * choices["C"][count] 
            for count in count_ranges["C"]
        ])
        
        return n_sum <= c_sum // 2 + 1

    custom_rules = [rule_DBE_onehot, rule_oc_ratio_onehot]#, rule_nitrogen_limit_onehot]


    # Test with a specific mass
    measured_mass_user = 285.136493
    error_user = 2  # ppm

    print(f"\n--- Decomposing mass {measured_mass_user} +/- {error_user} ppm (One-Hot method) ---")
    
    # Single run with detailed profiling
    print("\n--- SINGLE RUN WITH DETAILED PROFILING ---")
    solutions_onehot = None
    solutions_onehot = solve_mass_decomposition(
        measured_mass_user,
        error_user,
        elements_data_generic,
        rules=custom_rules,
        max_solutions=50,  # Reduced for detailed profiling
        enable_profiling=True,
    )
    
    if solutions_onehot:
        print(f"\nFound {len(solutions_onehot)} possible formulas:")
        for i, formula in enumerate(solutions_onehot[:10]):  # Show first 10
            formula_str = "".join([f"{el}{count}" for el, count in sorted(formula.items())])
            mass = sum(
                elements_data_generic[el]["mass"] * count
                for el, count in formula.items()
            )
            delta_ppm = ((mass - measured_mass_user) / measured_mass_user) * 1e6
            print(f"  {i + 1}. {formula_str} (Mass: {mass:.6f}, Delta: {delta_ppm:.2f} ppm)")
        if len(solutions_onehot) > 10:
            print(f"  ... and {len(solutions_onehot) - 10} more solutions")
    else:
        print(f"No solutions found for mass {measured_mass_user}.")
    
   