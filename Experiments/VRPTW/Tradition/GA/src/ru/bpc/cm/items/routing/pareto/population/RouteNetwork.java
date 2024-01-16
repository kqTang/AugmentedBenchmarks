package ru.bpc.cm.items.routing.pareto.population;

import java.util.ArrayList;

import ru.bpc.cm.items.routing.pareto.problem.Problem;

class RouteNetwork extends ArrayList<Route> {

	private static final long serialVersionUID = -6180881689433423705L;

	RouteNetwork() {
		super();
	}

	RouteNetwork(int capacity) {
		super(capacity);
	}

	int evaluateTotalDistance() {
		int tc = 0;
		for (Route route : this) {
			tc += route.getDistance();
		}
		return tc;
	}

	private int evaluateTotalDistanceExcept(int routeIndex) {
		int tc = 0;
		for (int i = 0; i < this.size(); i++) {
			if (i != routeIndex) {
				tc += get(i).getDistance();
			}
		}
		return tc;
	}

	boolean removeCustomer(int customer) {
		for (Route route : this) {
			if (route.remove((Object) customer)) {
				if (route.isEmpty()) {
					remove(route);
				}
				return true;
			}
		}
		return false;
	}

	void insertCustomer(int customer) {
		int bestRouteIndex = -1;
		int bestIndex = -1;
		int bestCost = Integer.MAX_VALUE;
		int currentCost;
		for (int routeIndex = 0; routeIndex < this.size(); routeIndex++) {
			Route route = this.get(routeIndex);
			for (int index = 0; index <= route.size(); index++) {
				if (route.isFeasible(index, customer)) {
					currentCost = this.evaluateTotalDistanceExcept(routeIndex)
							+ route.getImaginaryDistance(index, customer);
					if (currentCost < bestCost) {
						bestRouteIndex = routeIndex;
						bestIndex = index;
						bestCost = currentCost;
					}
				}
			}
		}
		if (bestIndex == -1) {
			Route newRoute = new Route(Problem.getInstance().customersNumber);
			newRoute.add(customer);
			this.add(newRoute);
		} else {
			this.get(bestRouteIndex).add(bestIndex, customer);
		}
	}

}
