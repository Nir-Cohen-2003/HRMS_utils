from z3 import Solver, Bool, Sum, If, Int, Or, And, Not, sat, Optimize,is_true
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

def z3_solve_mass_decomposition(
    target_mass: float,
    error_ppm: float,
    element_details: Dict[str, Dict[str, float]],
    rules: list = None,
    max_solutions: int = 100,
    solver_options: dict = None,
    enable_profiling: bool = False,
):
    """
    Z3-based version of mass decomposition using binary variables
    for each possible count of each element.
    """
    if rules is None:
        rules = []
    if solver_options is None:
        solver_options = {}

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
    
    # Create Z3 solver
    solver = Solver()

    # Create one-hot binary variables for each element and possible count
    choices = {}  # choices[element][count] = binary variable
    count_ranges = {}  # Store valid count ranges for each element
    
    total_variables = 0
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        max_count = element_data.get("max", max_allowed_count)
        max_count = min(max_count, max_allowed_count)  # make sure we won't consider counts that would exceed max_mass
        count_ranges[element_symbol] = range(min_count, max_count + 1)
        choices[element_symbol] = {}
        
        for count in count_ranges[element_symbol]:
            choices[element_symbol][count] = Bool(f"Choice_{element_symbol}_{count}")
            total_variables += 1
    
    print(f"PROFILE: Created {total_variables} binary variables")
    timing_profile["variable_creation"] = perf_counter() - var_creation_start

    # Constraint creation phase
    constraint_start = perf_counter()

    # Constraint: exactly one count must be chosen for each element
    for element_symbol in element_symbols:
        solver.add(Sum([If(choices[element_symbol][count], 1, 0) 
                       for count in count_ranges[element_symbol]]) == 1)

    # Mass constraint - use scaled integers to avoid floating point issues
    scale_factor = 1000000  # Scale to 6 decimal places
    
    mass_contributions = []
    for element_symbol in element_symbols:
        for count in count_ranges[element_symbol]:
            scaled_mass = int(element_details[element_symbol]["mass"] * scale_factor)
            mass_contributions.append(
                If(choices[element_symbol][count], count * scaled_mass, 0)
            )
    
    total_mass_scaled = Sum(mass_contributions)
    min_mass_scaled = int(min_mass * scale_factor)
    max_mass_scaled = int(max_mass * scale_factor)
    
    # Add mass bounds as constraints
    solver.add(total_mass_scaled >= min_mass_scaled)
    solver.add(total_mass_scaled <= max_mass_scaled)
    
    timing_profile["constraint_creation"] = perf_counter() - constraint_start
    print(f"DEBUG: Mass constraints set to [{min_mass:.6f}, {max_mass:.6f}]")

    # Rule application phase
    rule_start = perf_counter()
    
    # Add heuristic rules if provided
    if rules:
        for i, rule_func in enumerate(rules):
            try:
                # Check if rule function expects one-hot parameters
                import inspect
                sig = inspect.signature(rule_func)
                if len(sig.parameters) == 2:
                    # New one-hot rule format for Z3
                    constraint = rule_func(choices, count_ranges)
                else:
                    # Create auxiliary variables for legacy rules
                    element_counts = {}
                    for element_symbol in element_symbols:
                        element_counts[element_symbol] = Sum([
                            If(choices[element_symbol][count], count, 0)
                            for count in count_ranges[element_symbol]
                        ])
                    constraint = rule_func(element_counts)
                
                if constraint is not None:
                    solver.add(constraint)
                    print(f"DEBUG: Added rule {i}")
            except Exception as e:
                print(f"Warning: Error applying rule {i}: {e}. Skipping rule.")
    
    timing_profile["rule_application"] = perf_counter() - rule_start

    # Solving loop
    solution_count = 0
    total_solving_time = 0.0
    total_extraction_time = 0.0
    total_exclusion_time = 0.0
    
    while solution_count < max_solutions:
        # Solve the problem
        solve_start = perf_counter()
        result = solver.check()
        solve_time = perf_counter() - solve_start
        total_solving_time += solve_time
        
        if enable_profiling:
            print(f"PROFILE: Solve iteration {solution_count + 1} took {solve_time:.4f} seconds")
        
        if result == sat:
            # Extract the solution and verify mass constraint
            extraction_start = perf_counter()
            
            model = solver.model()
            current_formula = {}
            actual_mass = 0.0
            
            for element_symbol in element_symbols:
                element_count = 0
                choices_found = 0
                for count in count_ranges[element_symbol]:
                    choice_value = model[choices[element_symbol][count]]
                    if is_true(choice_value):
                        element_count = count
                        choices_found += 1
                
                # Verify one-hot constraint
                if choices_found != 1:
                    print(f"ERROR: Element {element_symbol} has {choices_found} choices set to 1!")
                    # Debug: show all choice values for this element
                    for count in count_ranges[element_symbol]:
                        val = model[choices[element_symbol][count]]
                        if is_true(val):
                            print(f"  {element_symbol}[{count}] = True")
                
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
                break
                
            print(f"solution {solution_count + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
            found_formulas.append(current_formula)
            solution_count += 1
            
            # Add constraint to exclude this exact solution
            exclusion_start = perf_counter()
            
            current_solution_vars = []
            for element_symbol in element_symbols:
                for count in count_ranges[element_symbol]:
                    if is_true(model[choices[element_symbol][count]]):
                        current_solution_vars.append(choices[element_symbol][count])
            
            # Constraint: at least one of these variables must be False in future solutions
            if current_solution_vars:
                solver.add(Sum([If(var, 1, 0) for var in current_solution_vars]) <= len(current_solution_vars) - 1)
            
            exclusion_time = perf_counter() - exclusion_start
            total_exclusion_time += exclusion_time
            
        else:
            print(f"Z3 result: {result}. No more solutions found.")
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

# Z3-compatible rule functions
def rule_DBE_z3(choices, count_ranges):
    """
    DBE (Double Bond Equivalent) rule: H <= 2*C + 3
    Z3 version using If expressions
    """
    if "H" not in choices or "C" not in choices:
        return None
    
    h_sum = Sum([If(choices["H"][count], count, 0) for count in count_ranges["H"]])
    c_sum = Sum([If(choices["C"][count], count, 0) for count in count_ranges["C"]])
    
    return h_sum <= 2 * c_sum + 3

def rule_oc_ratio_z3(choices, count_ranges):
    """
    Oxygen-Carbon ratio rule: O <= C
    Z3 version using If expressions
    """
    if "O" not in choices or "C" not in choices:
        return None
    
    o_sum = Sum([If(choices["O"][count], count, 0) for count in count_ranges["O"]])
    c_sum = Sum([If(choices["C"][count], count, 0) for count in count_ranges["C"]])
    
    return o_sum <= c_sum

def rule_nitrogen_limit_z3(choices, count_ranges):
    """
    Nitrogen limit rule: N <= C/2 + 1
    Z3 version using If expressions
    """
    if "N" not in choices or "C" not in choices:
        return None
    
    n_sum = Sum([If(choices["N"][count], count, 0) for count in count_ranges["N"]])
    c_sum = Sum([If(choices["C"][count], count, 0) for count in count_ranges["C"]])
    
    return n_sum <= c_sum / 2 + 1

if __name__ == "__main__":
    # --- Example Usage ---
    print("Running Mass Decomposition Example with Z3 Solver...")

    # Define elements and their properties (same as original)
    elements_data_generic = {
        "C": {"mass": 12.000000, "min": 0, "max": 20},
        "H": {"mass": 1.007825, "min": 0, "max": 100},
        "O": {"mass": 15.994914, "min": 0, "max": 20},
        "N": {"mass": 14.003074, "min": 0, "max": 10},
        "S": {"mass": 31.972071, "min": 0, "max": 0},
        "P": {"mass": 30.973762, "min": 0, "max": 5},
        "F": {"mass": 18.998403, "min": 0, "max": 20},
        "Cl": {"mass": 34.968853, "min": 0, "max": 0},
        "Br": {"mass": 78.918337, "min": 0, "max": 0},
        "I": {"mass": 126.904473, "min": 0, "max": 2},
    }

    # Z3-compatible rules
    custom_rules = [rule_DBE_z3, rule_oc_ratio_z3]

    # Test with the same mass
    measured_mass_user = 285.136493
    error_user = 2  # ppm

    print(f"\n--- Decomposing mass {measured_mass_user} +/- {error_user} ppm (Z3 method) ---")
    
    # Single run with detailed profiling
    print("\n--- SINGLE RUN WITH DETAILED PROFILING ---")
    solutions_z3 = z3_solve_mass_decomposition(
        measured_mass_user,
        error_user,
        elements_data_generic,
        rules=custom_rules,
        max_solutions=50,
        enable_profiling=True,
    )
    
    if solutions_z3:
        print(f"\nFound {len(solutions_z3)} possible formulas:")
        for i, formula in enumerate(solutions_z3[:10]):  # Show first 10
            formula_str = "".join([f"{el}{count}" for el, count in sorted(formula.items())])
            mass = sum(
                elements_data_generic[el]["mass"] * count
                for el, count in formula.items()
            )
            delta_ppm = ((mass - measured_mass_user) / measured_mass_user) * 1e6
            print(f"  {i + 1}. {formula_str} (Mass: {mass:.6f}, Delta: {delta_ppm:.2f} ppm)")
        if len(solutions_z3) > 10:
            print(f"  ... and {len(solutions_z3) - 10} more solutions")
    else:
        print(f"No solutions found for mass {measured_mass_user}.")

