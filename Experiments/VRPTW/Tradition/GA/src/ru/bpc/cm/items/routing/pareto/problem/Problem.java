package ru.bpc.cm.items.routing.pareto.problem;

import java.util.ArrayList;
import java.util.List;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.RiderTimeWindow;

/**
 * @author mors741
 * Using singleton pattern
 */
public class Problem {
	private static Problem INSTANCE;

	/**
	 * Amount of customers in problem.
	 */
	public int customersNumber;
	/**
	 * Maximum amount of money that car can store.
	 */
	public final int vehicleCapacity;
	public final Point depot;
	public final int[][] distanceCoeffs;
	public final int[][] timeCoeffs;
	public final List<Point> customers; // First cusomer's id = 0, not 1!
	public final int maxRouteLength;
	/**
	 * Maximum amount of cars that can be used in solution.
	 */
	public final int maxCars;

	/**
	 * Maximum time that driver has.
	 */
	public final int maxRouteTime;
	/**
	 * The price of servicing one ATM.
	 */
	public final double servicePrice;
	/**
	 * The price of one kilometer of the route.
	 */
	public final double kmPrice;
	/**
	 * Maximum amount of ATMs in one route.
	 */
	public final int maxCustInRoute;

	/**
	 * Currency of all cash parameters.
	 */
	public final int currCode;
	/**
	 * ATM window mode. Normal and default (in which, each time window is placed
	 * on the maximum possible period);
	 */
	public final int windowMode;

	/**
	 * Time windows of car drivers.
	 * 
	 * @Unsupported
	 */
	public final ArrayList<RiderTimeWindow> riderTimeWindows;
	/**
	 * Max amount of cassettes that one car can store.
	 * 
	 * @Unsupported
	 */
	public final int volumeOneCar;
	/**
	 * Amount of cassettes that each ATM needs.
	 * 
	 * @Unsupported
	 */
	public final int[] amountOfCassettes;
	/**
	 * The price of servicing each ATM.
	 * 
	 * @Unsupported
	 */
	public final double[] atmPrice;
	
	private boolean isCorrect = true;

	private Problem(Matrix m) {
		vehicleCapacity = m.MaxMoney == 0 ? Integer.MAX_VALUE : m.MaxMoney;
		distanceCoeffs = m.distanceCoeffs;
		timeCoeffs = m.timeCoeffs; // in seconds
		customersNumber = distanceCoeffs.length - 1;

		maxRouteLength = m.MaxLength == 0 ? Integer.MAX_VALUE : m.MaxLength;
		maxCars = m.maxCars == 0 ? Integer.MAX_VALUE : m.maxCars;
		maxCustInRoute = m.MaxATMInWay == 0 ? Integer.MAX_VALUE : m.MaxATMInWay;
		maxRouteTime = m.MaxTime == 0 ? Integer.MAX_VALUE : m.MaxTime;

		depot = new Point(-1, m.ENC[0], m.depot, m.amountOfMoney[0], m.getTimeWindow(0).StartWork,
				m.getTimeWindow(0).EndWork, m.serviceTime[0]);
		customers = new ArrayList<Point>(customersNumber);
		

		for (int i = 1; i <= customersNumber; i++) {
			int startWork = m.getTimeWindow(i).StartWork;
			int endWork = m.getTimeWindow(i).EndWork;
			// if customer cann't be reached within time windows
			if (timeCoeffs[0][i] > m.getTimeWindow(i).EndWork){
				// move time window forward
				int delta = timeCoeffs[0][i] - startWork;
				startWork += delta;
				endWork += delta;
				// and mark as INCORRECT
				isCorrect = false;
				
			}
			customers.add(new Point(i - 1, m.ENC[i], m.ATM[i], m.amountOfMoney[i],
					startWork, endWork, m.serviceTime[i]));
		}

		riderTimeWindows = m.getRiderTimeWindows();
		amountOfCassettes = m.amountOfCassettes;
		volumeOneCar = m.VolumeOneCar;
		servicePrice = m.FixPrice;
		kmPrice = m.LengthPrice;
		atmPrice = m.AtmPrice;
		currCode = m.currCode;
		windowMode = m.windowMode;
	}

	public static Problem getInstance() {
		if (INSTANCE == null) {
			throw new NullPointerException("Use Problem.initialize(Matrix m); first.");
		} else {
			return INSTANCE;
		}
	}

	public static void initialize(Matrix m) {
		INSTANCE = new Problem(m);
	}

	public Point getDepot() {
		return depot;
	}

	public int distanceBetween(Point p1, Point p2) {
		return distanceCoeffs[p1.getId() + 1][p2.getId() + 1];
	}

	public int timeBetween(Point p1, Point p2) {
		return timeCoeffs[p1.getId() + 1][p2.getId() + 1];
	}

	public Point getCustomer(int i) {
		return customers.get(i);
	}

	public boolean isCorrect() {
		return isCorrect;
	}
}
