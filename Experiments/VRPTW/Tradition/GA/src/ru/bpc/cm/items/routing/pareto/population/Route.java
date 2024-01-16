package ru.bpc.cm.items.routing.pareto.population;

import java.util.ArrayList;

import ru.bpc.cm.items.routing.pareto.common.ArrayShuffler;
import ru.bpc.cm.items.routing.pareto.problem.Point;
import ru.bpc.cm.items.routing.pareto.problem.Problem;

class Route extends ArrayList<Integer> {

	private static final long serialVersionUID = 1L;

	private int[] invertArray = { -2, -1, 1, 2 };

	Route(int customersNumber) {
		super(customersNumber);
	}

	Route(Route route) {
		super(route);
	}

	void invert(int startPoz) {
		ArrayShuffler.shuffle(invertArray);

		for (int i = 0; i < 4; i++) {
			if (startPoz + invertArray[i] < this.size() && startPoz + invertArray[i] >= 0) {
				int temp = this.get(startPoz);
				this.set(startPoz, this.get(startPoz + invertArray[i]));
				this.set(startPoz + invertArray[i], temp);
				break;
			}
		}
	}

	int getDistance() {
		int res = Problem.getInstance().getDepot().distanceTo(Problem.getInstance().getCustomer(this.get(0)));
		int i;
		for (i = 0; i < this.size() - 1; i++) {
			res += Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getCustomer(this.get(i + 1)));
		}
		return res + Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getDepot());
	}

	int getTime() {
		int pointTime = Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().getServiceTime(); // start time of depot
		pointTime -= Problem.getInstance().getDepot().getServiceTime(); // if not serving depot in the beginning
		int prevCust = 0;
		for (Integer pid : this) {
			pointTime = Math.max(pointTime + Problem.getInstance().getCustomer(prevCust).timeTo(Problem.getInstance().getCustomer(pid))
					+ Problem.getInstance().getCustomer(prevCust).getServiceTime(), Problem.getInstance().getCustomer(pid).getReadyTime());
			prevCust = pid;
		}
		pointTime += Problem.getInstance().getCustomer(prevCust).timeTo(Problem.getInstance().getDepot());
		return pointTime;
	}

	int getImaginaryDistance(int index, int cust) {
		int res;
		if (index == 0) {
			res = Problem.getInstance().getDepot().distanceTo(Problem.getInstance().getCustomer(cust));
			res += Problem.getInstance().getCustomer(cust).distanceTo(Problem.getInstance().getCustomer(this.get(0)));
			int i;
			for (i = 0; i < this.size() - 1; i++) {
				res += Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getCustomer(this.get(i + 1)));
			}
			return res + Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getDepot());
		} else {
			res = Problem.getInstance().getDepot().distanceTo(Problem.getInstance().getCustomer(this.get(0)));
			int i;
			for (i = 0; i < index - 1; i++) {
				res += Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getCustomer(this.get(i + 1)));
			}

			res += Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getCustomer(cust));
			if (index != this.size()) {
				res += Problem.getInstance().getCustomer(cust).distanceTo(Problem.getInstance().getCustomer(this.get(i + 1)));
			} else {
				return res + Problem.getInstance().getCustomer(cust).distanceTo(Problem.getInstance().getDepot());
			}

			for (i = index; i < this.size() - 1; i++) {
				res += Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getCustomer(this.get(i + 1)));
			}
			return res + Problem.getInstance().getCustomer(this.get(i)).distanceTo(Problem.getInstance().getDepot());
		}

	}

	public Route clone() {
		Route clone = new Route(this);
		return clone;
	}

	boolean isFeasible() {
		// Check capacity constrain
		int capacity = 0;
		for (int gene : this) {
			capacity += Problem.getInstance().getCustomer(gene).getDemand();
		}
		if (capacity > Problem.getInstance().vehicleCapacity) {
			return false;
		}
		// Check time windows constrain
		int time = Math.max(
				Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().timeTo(Problem.getInstance().getCustomer(this.get(0))),
				Problem.getInstance().getCustomer(this.get(0)).getReadyTime());
		for (int i = 0; i < this.size(); i++) {
			Point customer = Problem.getInstance().getCustomer(this.get(i));

			if (i != 0) {
				time += Problem.getInstance().getCustomer(this.get(i - 1)).timeTo(customer);
			}

			if (time > customer.getDueDate()) {
				return false;
			} else if (time < customer.getReadyTime()) {
				time = customer.getReadyTime();
			}

			time += customer.getServiceTime();
		}

		if (time + Problem.getInstance().getCustomer(this.getLastGene()).timeTo(Problem.getInstance().getDepot()) > Problem.getInstance().getDepot()
				.getDueDate()) {
			return false;
		}

		return true;
	}

	boolean isFeasible(int index, int cust) {
		// Check capacity constrain
		int capacity = 0;
		for (int gene : this) {
			capacity += Problem.getInstance().getCustomer(gene).getDemand();
		}
		capacity += Problem.getInstance().getCustomer(cust).getDemand();
		if (capacity > Problem.getInstance().vehicleCapacity) {
			return false;
		}
		// Check time windows constrain
		int time;
		Point customer;
		if (index == 0) {
			time = Math.max(Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().timeTo(Problem.getInstance().getCustomer(cust)),
					Problem.getInstance().getCustomer(cust).getReadyTime());
			customer = Problem.getInstance().getCustomer(cust);

			if (time > customer.getDueDate()) {
				return false;
			}

			time += customer.getServiceTime();

			customer = Problem.getInstance().getCustomer(this.get(0));
			time += Problem.getInstance().getCustomer(cust).timeTo(customer);
			if (time > customer.getDueDate()) {
				return false;
			} else if (time < customer.getReadyTime()) {
				time = customer.getReadyTime();
			}

			time += customer.getServiceTime();

			for (int i = 1; i < this.size(); i++) {
				customer = Problem.getInstance().getCustomer(this.get(i));
				time += Problem.getInstance().getCustomer(this.get(i - 1)).timeTo(customer);
				if (time > customer.getDueDate()) {
					return false;
				} else if (time < customer.getReadyTime()) {
					time = customer.getReadyTime();
				}

				time += customer.getServiceTime();
			}

		} else {
			time = Math.max(
					Problem.getInstance().getDepot().getReadyTime() + Problem.getInstance().getDepot().timeTo(Problem.getInstance().getCustomer(this.get(0))),
					Problem.getInstance().getCustomer(this.get(0)).getReadyTime());
			customer = Problem.getInstance().getCustomer(this.get(0));
			if (time > customer.getDueDate()) {
				return false;
			}
			time += customer.getServiceTime();
			int i;
			for (i = 1; i < index; i++) { // Before index
				customer = Problem.getInstance().getCustomer(this.get(i));

				time += Problem.getInstance().getCustomer(this.get(i - 1)).timeTo(customer);

				if (time > customer.getDueDate()) {
					return false;
				} else if (time < customer.getReadyTime()) {
					time = customer.getReadyTime();
				}

				time += customer.getServiceTime();
			}

			// index
			customer = Problem.getInstance().getCustomer(cust);
			time += Problem.getInstance().getCustomer(this.get(i - 1)).timeTo(customer);
			if (time > customer.getDueDate()) {
				return false;
			} else if (time < customer.getReadyTime()) {
				time = customer.getReadyTime();
			}

			time += customer.getServiceTime();

			// index + 1
			if (index != this.size()) {
				customer = Problem.getInstance().getCustomer(this.get(index));
				time += Problem.getInstance().getCustomer(cust).timeTo(customer);
				if (time > customer.getDueDate()) {
					return false;
				} else if (time < customer.getReadyTime()) {
					time = customer.getReadyTime();
				}

				time += customer.getServiceTime();

				// other
				for (i = index + 1; i < this.size(); i++) {
					customer = Problem.getInstance().getCustomer(this.get(i));

					time += Problem.getInstance().getCustomer(this.get(i - 1)).timeTo(customer);

					if (time > customer.getDueDate()) {
						return false;
					} else if (time < customer.getReadyTime()) {
						time = customer.getReadyTime();
					}

					time += customer.getServiceTime();
				}
			}
		}
		if (index == this.size()) {// on the last place
			if (time + Problem.getInstance().getCustomer(cust).timeTo(Problem.getInstance().getDepot()) > Problem.getInstance().getDepot().getDueDate()) {
				return false;
			}
		} else {
			if (time + Problem.getInstance().getCustomer(this.getLastGene()).timeTo(Problem.getInstance().getDepot()) > Problem.getInstance().getDepot()
					.getDueDate()) {
				return false;
			}
		}
		return true;
	}

	int getLastGene() {
		return this.get(this.size() - 1);
	}
}
