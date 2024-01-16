package ru.bpc.cm.items.routing.pareto.population;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import ru.bpc.cm.items.routing.pareto.Pareto;
import ru.bpc.cm.items.routing.pareto.problem.Point;
import ru.bpc.cm.items.routing.pareto.problem.Problem;
/**
 * 
 * @author mors741
 * Using static factory pattern 
 */
class IndividualFactory {

	/**
	 * Generate chromosome using random permutations
	 */
	static Individual generateRandomChromosome() {
		Individual indiv = new Individual();
		// Filling in the order
		for (int i = 0; i < Problem.getInstance().customersNumber; i++) {
			indiv.getChromosome().add(i);
		}
		// And shuffling
		Collections.shuffle(indiv.getChromosome());
		return indiv;
	}

	/**
	 * Generate chromosome using greedy procedure from 22p.
	 */
	static Individual generateGreedyChromosome() {
		Individual indiv = new Individual();
		Random random = new Random();
		List<Integer> tempChromosome = new ArrayList<Integer>(Problem.getInstance().customersNumber);
		for (int i = 0; i < Problem.getInstance().customersNumber; i++) {
			tempChromosome.add(i);
		}

		while (!tempChromosome.isEmpty()) {
			int randIndex = (int) (random.nextInt(tempChromosome.size()));
			int randCustId = tempChromosome.get(randIndex);
			tempChromosome.remove(randIndex);
			indiv.getChromosome().add(randCustId);
			Point randCust = Problem.getInstance().getCustomer(randCustId);
			while (true) {
				// Finding the nearest point
				double dist;
				double minDist = Pareto.EUCLIDEAN_RADIUS; // If the distance is
															// greater than
															// EUCLIDEAN_RADIUS,
															// then this point
															// is not
															// interesting
															// anyway
				int nearestId = -1;
				for (Point p : Problem.getInstance().customers) {
					dist = randCust.distanceTo(p);
					if (dist < minDist && dist > 0 && // To exclude itself
							!indiv.getChromosome().contains(p.getId())) { // Still
																			// not
																			// in
																			// list
						minDist = dist;
						nearestId = p.getId();
					}
				}
				if (nearestId == -1) { // The nearest point is not exist
					break;
				}
				tempChromosome.remove((Object) nearestId);
				indiv.getChromosome().add(nearestId);
				randCust = Problem.getInstance().getCustomer(nearestId);
			}
		}
		return indiv;
	}
}
