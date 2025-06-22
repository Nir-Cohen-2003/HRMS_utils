import pulp
import math
from typing import List, Dict

def solve_mass_decomposition_onehot(
    target_mass: float,
    error_ppm: float,
    element_details: Dict[str, Dict[str, float]],
    rules: list = None,
    max_solutions: int = 100,
    solver_options: dict = None,
):
    """
    One-hot encoding version of mass decomposition using binary variables
    for each possible count of each element.
    """
    if rules is None:
        rules = []
    if solver_options is None:
        solver_options = {"msg": False}

    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute

    print(f"Target mass: {target_mass:.6f}, Error: {error_ppm} ppm")
    print(f"Searching in mass range: [{min_mass:.6f} - {max_mass:.6f}]")

    found_formulas = []
    element_symbols = list(element_details.keys())

    # Create problem
    prob = pulp.LpProblem("MassDecomposition_OneHot")

    # Create one-hot binary variables for each element and possible count
    choices = {}  # choices[element][count] = binary variable
    count_ranges = {}  # Store valid count ranges for each element
    
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        default_max = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        max_count = min(element_data.get("max", default_max), 100)  # Cap at 100 for performance
        
        count_ranges[element_symbol] = range(min_count, max_count + 1)
        choices[element_symbol] = {}
        
        for count in count_ranges[element_symbol]:
            choices[element_symbol][count] = pulp.LpVariable(
                f"Choice_{element_symbol}_{count}", cat="Binary"
            )

    # Constraint: exactly one count must be chosen for each element
    for element_symbol in element_symbols:
        prob += pulp.lpSum([
            choices[element_symbol][count] 
            for count in count_ranges[element_symbol]
        ]) == 1, f"OneCount_{element_symbol}"

    # Create auxiliary variables for actual element counts (this makes debugging easier)
    element_counts = {}
    for element_symbol in element_symbols:
        element_counts[element_symbol] = pulp.lpSum([
            choices[element_symbol][count] * count
            for count in count_ranges[element_symbol]
        ])

    # Mass constraint using the auxiliary variables
    total_mass = pulp.lpSum([
        element_counts[element_symbol] * element_details[element_symbol]["mass"]
        for element_symbol in element_symbols
    ])
    
    prob += total_mass >= min_mass, "MinMassConstraint"
    prob += total_mass <= max_mass, "MaxMassConstraint"
    
    print(f"DEBUG: Mass constraints set to [{min_mass:.6f}, {max_mass:.6f}]")

    # Add heuristic rules if provided
    if rules:
        # Apply rules using the count variables
        for i, rule_func in enumerate(rules):
            try:
                constraint = rule_func(element_counts)
                if constraint is not None:
                    prob += constraint, f"HeuristicRule_{i}"
                    print(f"DEBUG: Added rule {i}")
            except Exception as e:
                print(f"Warning: Error applying rule {i}: {e}. Skipping rule.")

    # Open output file for writing solutions
    solution_count = 0
    
    while solution_count < max_solutions:
        # Solve the problem
        try:
            solver = pulp.GUROBI(**solver_options)
        except Exception:
            solver = pulp.PULP_CBC_CMD(**solver_options)

        status = prob.solve(solver)
        
        if pulp.LpStatus[status] == "Optimal":
            # Extract the solution and verify mass constraint
            current_formula = {}
            actual_mass = 0.0
            
            for element_symbol in element_symbols:
                for count in count_ranges[element_symbol]:
                    if pulp.value(choices[element_symbol][count]) == 1:
                        if count > 0:
                            current_formula[element_symbol] = count
                        actual_mass += count * element_details[element_symbol]["mass"]
                        break
            
            # Double-check mass constraint
            delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
            
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
            
            found_formulas.append(current_formula)
            solution_count += 1
            
            # Display solution details
            formula_str = "".join([f"{el}{count}" for el, count in sorted(current_formula.items())])
            print(f"Solution {solution_count}: {formula_str} (Mass: {actual_mass:.6f}, Delta: {delta_ppm:.2f} ppm)")
            
            # Add constraint to exclude this exact solution (Sudoku-style)
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
        else:
            print(f"Solver status: {pulp.LpStatus[status]}. No more solutions found.")
            break

    return found_formulas


# ...existing code...

if __name__ == "__main__":
    # --- Example Usage ---
    print("Running Mass Decomposition Example with One-Hot Encoding...")

    # Define elements and their properties
    elements_data_generic = {
        "C": {"mass": 12.000000, "min": 0, "max": 20},
        "H": {"mass": 1.007825, "min": 0, "max": 50},
        "O": {"mass": 15.994914, "min": 0, "max": 20},
        "N": {"mass": 14.003074, "min": 0, "max": 10},
    }

    # Define heuristic rules
    def rule_hydrogen_count(vars_d):
        # H <= 2*C + 3
        if "H" in vars_d and "C" in vars_d:
            return vars_d["H"] <= 2 * vars_d["C"] + 3
        return None

    def rule_oc_ratio(vars_d):
        # O <= C
        if "O" in vars_d and "C" in vars_d:
            return vars_d["O"] <= vars_d["C"]
        return None

    custom_rules = [rule_hydrogen_count, rule_oc_ratio]

    # Test with a specific mass
    measured_mass_user = 285.136493
    error_user = 200  # ppm

    print(f"\n--- Decomposing mass {measured_mass_user} +/- {error_user} ppm (One-Hot method) ---")
    solutions_onehot = solve_mass_decomposition_onehot(
        measured_mass_user,
        error_user,
        elements_data_generic,
        rules=custom_rules,
        max_solutions=500,
    )
    
    if solutions_onehot:
        print(f"\nFound {len(solutions_onehot)} possible formulas:")
        for i, formula in enumerate(solutions_onehot):
            formula_str = "".join([f"{el}{count}" for el, count in sorted(formula.items())])
            mass = sum(
                elements_data_generic[el]["mass"] * count
                for el, count in formula.items()
            )
            delta_ppm = ((mass - measured_mass_user) / measured_mass_user) * 1e6
            print(f"  {i + 1}. {formula_str} (Mass: {mass:.6f}, Delta: {delta_ppm:.2f} ppm)")
    else:
        print(f"No solutions found for mass {measured_mass_user}.")