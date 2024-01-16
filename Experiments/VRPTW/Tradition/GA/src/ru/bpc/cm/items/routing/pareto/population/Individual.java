package ru.bpc.cm.items.routing.pareto.population;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ru.bpc.cm.items.routing.SolutionRoutes;
import ru.bpc.cm.items.routing.SolutionStatus;
import ru.bpc.cm.items.routing.pareto.problem.Point;
import ru.bpc.cm.items.routing.pareto.problem.Problem;

/**
 * @author EAH
 *
 */
public class Individual implements SolutionRoutes, Comparable<Individual> {

	private List<Integer> chromosome;
	private RouteNetwork routes;
	private int routesNumber = 1;
	private int totalCost = 0;
	private int isFeasible = 0;
	private int paretoRank;
	private SolutionStatus status = SolutionStatus.CONSTRAINTS_OK;

	Individual() {
		chromosome = new ArrayList<Integer>(Problem.getInstance().customersNumber);
		routes = new RouteNetwork(Problem.getInstance().customersNumber);
	}

	void mutate() {
		if (this.paretoRank != 1) {
			routesNumber = routes.size();
			List<Integer> feasibleRoutes = new ArrayList<Integer>(routesNumber);
			for (int i = 0; i < routesNumber; i++) {
				if (routes.get(i).size() > 1) {
					feasibleRoutes.add(i);
				}
			}
			Random rnd = new Random();

			Route route = this.routes.get(feasibleRoutes.get(rnd.nextInt(feasibleRoutes.size())));
			route.invert(rnd.nextInt(route.size()));
		}
	}

	void evaluateRoutes() {
		// Phase 1
		int currentCapacity = Problem.getInstance().vehicleCapacity;
		// For first customer
		int gene = chromosome.get(0);
		Point customer = Problem.getInstance().getCustomer(gene);
		int currentTime = Math.max(
				Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().timeTo(customer),
				customer.getReadyTime());
		routes.add(new Route(Problem.getInstance().customersNumber));
		currentCapacity -= customer.getDemand();

		if (currentCapacity < 0 // Not enough money
				|| currentTime > customer.getDueDate()) { // To late -> create new route

			currentCapacity = Problem.getInstance().vehicleCapacity - customer.getDemand();
			currentTime = Math.max(
					Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().timeTo(customer),
					customer.getReadyTime());

			routes.add(new Route(Problem.getInstance().customersNumber));
			routesNumber++;

		} else if (currentTime < customer.getReadyTime()) { // To early (wait to ReadyTime)
			currentTime = customer.getReadyTime();
		}

		currentTime += customer.getServiceTime();

		routes.get(routesNumber - 1).add(gene);
		int prevGene = gene;
		// For others
		for (int i = 1; i < chromosome.size(); i++) {
			gene = chromosome.get(i);
			customer = Problem.getInstance().getCustomer(gene);

			currentTime += Problem.getInstance().getCustomer(prevGene).timeTo(customer);

			currentCapacity -= customer.getDemand();

			if (currentCapacity < 0 // Not enough money
					|| currentTime > customer.getDueDate() // To late
					|| currentTime >= Problem.getInstance().maxRouteTime
					|| routes.get(routesNumber - 1).size() >= Problem.getInstance().maxCustInRoute
					|| routes.get(routesNumber - 1).getDistance() >= Problem.getInstance().maxRouteLength) { // create new route

				currentCapacity = Problem.getInstance().vehicleCapacity - customer.getDemand();
				currentTime = Math.max(Problem.getInstance().getDepot().getReadyTime()
						+ Problem.getInstance().getDepot().timeTo(customer), customer.getReadyTime());

				routes.add(new Route(Problem.getInstance().customersNumber));
				routesNumber++;

			} else if (currentTime < customer.getReadyTime()) { // To early (wait to ReadyTime)
				currentTime = customer.getReadyTime();
			}

			currentTime += customer.getServiceTime();

			routes.get(routesNumber - 1).add(gene);
			prevGene = gene;
		}

		// Phase 2
		for (int i = 1; i < routesNumber; i++) {
			int prevRouteLastIndex = routes.get(i - 1).size() - 1;
			int sumCost1 = routes.get(i - 1).getDistance() + routes.get(i).getDistance();
			routes.get(i).add(0, routes.get(i - 1).get(prevRouteLastIndex));
			routes.get(i - 1).remove(prevRouteLastIndex);

			if (routes.get(i - 1).isEmpty() || !routes.get(i).isFeasible()
					|| routes.get(i - 1).getDistance() + routes.get(i).getDistance() > sumCost1) {
				routes.get(i - 1).add(routes.get(i).get(0));
				routes.get(i).remove(0);
			}
		}

		totalCost = routes.evaluateTotalDistance();
	}

	private boolean theSameRoutesAs(Individual indiv) {

		if (this.routes.size() != indiv.routes.size()) {
			return false;
		}
		Route route;
		for (int i = 0; i < this.routes.size(); i++) {
			route = this.routes.get(i);
			if (route.size() != indiv.routes.get(i).size()) {
				return false;
			}
			for (int j = 0; j < route.size(); j++) {
				if (route.get(j) != indiv.routes.get(i).get(j)) {
					return false;
				}
			}
		}
		return true;
	}

	public Individual clone() {
		Individual clone = new Individual();
		clone.chromosome = new ArrayList<Integer>(this.chromosome);
		for (Route route : this.routes) {
			clone.routes.add(new Route(route));
		}
		clone.paretoRank = this.paretoRank;
		clone.totalCost = this.totalCost;
		clone.routesNumber = this.routesNumber;
		return clone;
	}

	boolean dominates(Individual other) {
		if (this.totalCost <= other.totalCost && this.routesNumber <= other.routesNumber
				&& (this.totalCost < other.totalCost || this.routesNumber < other.routesNumber)) {
			return true;
		} else {
			return false;
		}
	}

	void backToChromosome() {
		chromosome.clear();
		for (Route r : routes) {
			for (int gene : r) {
				chromosome.add(gene);
			}
		}
		routes.clear();
		routesNumber = 1;
		totalCost = 0;
		paretoRank = 0;
	}

	public String toString() {
		StringBuilder str = new StringBuilder(chromosome.size() * 2);
		str.append("<" + paretoRank + "> (" + routesNumber + "; " + totalCost + ") ");
		for (List<Integer> route : routes) {
			str.append("[");
			for (int gene : route) {
				str.append(gene + " ");
			}

			str.append("] ");
		}
		return str.toString();
	}

	public boolean equals(Object obj) {
		return this.theSameRoutesAs((Individual) obj);
	}

	RouteNetwork getRouteNetwork() {
		return routes;
	}

	List<Integer> getChromosome() {
		return this.chromosome;
	}

	Route getRoute(int i) {
		return routes.get(i);
	}

	public int getRoutesNumber() {
		return routesNumber;
	}

	int getIsFeasible() {
		return isFeasible;
	}

	public int getTotalCost() {
		return totalCost;
	}

	int getParetoRank() {
		return paretoRank;
	}

	void setParetoRank(int paretoRank) {
		this.paretoRank = paretoRank;
	}

	void setChromosome(List<Integer> newChromosome) {
		this.chromosome = new ArrayList<Integer>(newChromosome);
	}

	public ArrayList<? extends ArrayList<Integer>> getRoutes() {
		ArrayList<ArrayList<Integer>> aRoutes = new ArrayList<ArrayList<Integer>>();
		for (Route route : this.getRouteNetwork()) {
			Route aRoute = new Route(route.size());
			for (Integer point : route) {
				aRoute.add(point + 1);
			}
			aRoutes.add(aRoute);
		}
		return aRoutes;
	}
	
	@Override
	public int compareTo(Individual o) {
		return this.getRoutesNumber() - o.getRoutesNumber();
	}

	public SolutionStatus getStatus() {
		return status;
	}

	public void setStatus(SolutionStatus status) {
		this.status = status;
	}
}
