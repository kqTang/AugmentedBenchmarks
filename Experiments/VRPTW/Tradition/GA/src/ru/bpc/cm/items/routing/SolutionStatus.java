package ru.bpc.cm.items.routing;

public enum SolutionStatus {
	/**
	 * All problem constraints are satisfied
	 */
	CONSTRAINTS_OK(0),	
	
	/**
	 * Time windows constraint is violated
	 */
	TIME_CONSTRAINT_VIOLATION(1),
	
	/**
	 * Cars or ATMs amount constraint is violated
	 */
	CARS_OR_ATMS_CONSTRAINT_VIOLATION(2),
	
	/**
	 * Solution doesn't satisfy all ATM's needs
	 */
	SUMM_CONSTRAINT_VIOLATION(3);

	int value;

	SolutionStatus(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
}
