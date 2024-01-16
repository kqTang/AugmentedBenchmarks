package ru.bpc.cm.items.routing.pareto;

import java.util.ArrayList;
import java.util.List;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.SolutionStatus;
import ru.bpc.cm.items.routing.pareto.population.Individual;
import ru.bpc.cm.items.routing.pareto.population.Population;
import ru.bpc.cm.items.routing.pareto.problem.Problem;

public class Pareto {

	public static int POPULATION_SIZE;
	public static double CROSSOVER_RATE = 0.8;
	public static double MUTATION_RATE = 0.1;

	public static double INIT_RAND_RATE = 0.9;
	public static double EUCLIDEAN_RADIUS;

	public static double MAX_SAME_RESULT = 5;
	
	public static boolean CARS_OVER_COST = false;

	public static boolean SHOW_DEBUG = false;
	public static boolean SHOW_OPTIMAL = false;
	public static boolean SHOW_GEN = false;
	public static boolean SHOW_RESULT = false;

	private Population population;

	public Pareto(Matrix problem) {
		Problem.initialize(problem);
		int[][] distanceCoeffs = Problem.getInstance().distanceCoeffs;
		POPULATION_SIZE = 10 + 10 * (int) (0.004 * Math.pow((distanceCoeffs.length - 1), 2));
		int total = 0;
		for (int i = 0; i < distanceCoeffs.length; i++) {
			for (int j = 0; j < i; j++) {
				total += distanceCoeffs[i][j];
			}
		}
		EUCLIDEAN_RADIUS = (2 * total / (distanceCoeffs.length * (distanceCoeffs.length - 1)));
		population = new Population();
	}

	public Individual computeResult() {
		population.initialize();
		List<Individual> prevOptimalIndividualList = new ArrayList<Individual>();

		population.evaluateRoutes();
		population.determineParetoRanks();

		if (SHOW_DEBUG) {
			System.out.println(" ------------------------------------- 0 -------------------------------------");
			population.showInverse();
		}

		int generationCounter = 1;
		List<Individual> optimalIndividualList = null;
		boolean areListsIdentical;
		int sameResultCounter = 0;
		while (sameResultCounter < MAX_SAME_RESULT) {
			population.mate();
			population.mutation();
			population.backToChromosome();
			population.evaluateRoutes();
			population.determineParetoRanks();

			optimalIndividualList = population.getOptimalList();

			if (SHOW_DEBUG || SHOW_OPTIMAL) {
				System.out.println(" ------------------------------------- " + (generationCounter)
						+ " -------------------------------------");
			}
			if (SHOW_DEBUG) {
				population.showInverse();
			}
			if (SHOW_DEBUG || SHOW_OPTIMAL) {
				System.out.println("OPTIMAL: " + optimalIndividualList.size());
				for (Individual i : optimalIndividualList) {
					System.out.println(i);
				}
			}
			generationCounter++;

			areListsIdentical = true;
			if (optimalIndividualList.size() == prevOptimalIndividualList.size()) {
				for (int i = 0; i < optimalIndividualList.size(); i++) {
					if (optimalIndividualList.get(i).getRoutesNumber() != prevOptimalIndividualList.get(i)
							.getRoutesNumber()
							|| optimalIndividualList.get(i).getTotalCost() != prevOptimalIndividualList.get(i)
									.getTotalCost()) {
						areListsIdentical = false;
					}
				}
				if (areListsIdentical) {
					sameResultCounter++;
				} else {
					sameResultCounter = 0;
				}
			}
			prevOptimalIndividualList = optimalIndividualList;
			if (SHOW_DEBUG) {
				System.out.println("SAME: " + sameResultCounter);
			}
		}
		if (SHOW_DEBUG || SHOW_GEN) {
			System.out.println("TOTAL GENERATIONS = " + (generationCounter - 1));
		}
		if (SHOW_DEBUG || SHOW_OPTIMAL || SHOW_RESULT) {
			System.out.println("RESULT: " + optimalIndividualList.size());
			for (Individual i : optimalIndividualList) {
				System.out.println(i);
			}
		}
		
		Individual bestIndividual = null;
		
		if (!Problem.getInstance().isCorrect()) {
			bestIndividual = optimalIndividualList.get(optimalIndividualList.size()-1);
			System.out.println("STATUS = 1");
			bestIndividual.setStatus(SolutionStatus.TIME_CONSTRAINT_VIOLATION);
			return bestIndividual;
		}

		if (CARS_OVER_COST) {
			bestIndividual = optimalIndividualList.get(0);
			if (bestIndividual.getRoutesNumber() > Problem.getInstance().maxCars) {
				bestIndividual.setStatus(SolutionStatus.CARS_OR_ATMS_CONSTRAINT_VIOLATION);
			}
		} else {
			int tempLast = optimalIndividualList.size()-1;
			
			while (tempLast >= 0) {
				bestIndividual = optimalIndividualList.get(tempLast);
				if (bestIndividual.getRoutesNumber() <= Problem.getInstance().maxCars) {
					return bestIndividual;
				} else {	
					tempLast--;
				}
			}
			bestIndividual.setStatus(SolutionStatus.CARS_OR_ATMS_CONSTRAINT_VIOLATION);
			
		}
		

		return bestIndividual;
		
	}
}