package ru.bpc.cm.items.routing.pareto.problem;

import ru.bpc.cm.items.routing.pareto.Pareto;

public class Point {
	private final int id;
	private final int ENC;
	private final String ATM;
	private final int demand;
	private final int rtime;
	private final int ddate;
	private final int stime;

	Point(int id, int ENC, String ATM, int demand, int rtime, int ddate, int stime) {
		this.id = id;
		this.ENC = ENC;
		this.ATM = ATM;
		this.demand = demand;
		this.rtime = rtime;
		this.ddate = ddate;
		this.stime = stime;
	}

	public int getId() {
		return id;
	}

	public int getENC() {
		return ENC;
	}

	public String getATM() {
		return ATM;
	}

	public int getDemand() {
		return demand;
	}

	public int getReadyTime() {
		return rtime;
	}

	public int getDueDate() {
		return ddate;
	}

	public int getServiceTime() {
		return stime;
	}

	@Override
	public String toString() {
		return String.valueOf(this.id);
	}

	public int timeTo(Point p) {
		return Problem.getInstance().timeBetween(this, p);
	}

	public int distanceTo(Point p) {
		return Problem.getInstance().distanceBetween(this, p);
		// return Math.sqrt(Math.pow(this.y-p.y, 2)+Math.pow(this.x-p.x, 2));
	}

	/**
	 * @return the number of the nearest point within Euclidean area or -1, if
	 *         it's not exists
	 */
	public int getNearestCustomerId() {
		double dist;
		double minDist = Pareto.EUCLIDEAN_RADIUS;
		int minId = -1;
		for (Point p : Problem.getInstance().customers) {
			dist = this.distanceTo(p);
			if (dist < minDist && dist > 0) {
				minDist = dist;
				minId = p.getId();
			}
		}
		return minId;
	}
}
