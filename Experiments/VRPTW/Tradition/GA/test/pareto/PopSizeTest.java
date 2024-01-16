package pareto;

import org.junit.Test;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.pareto.Pareto;
import ru.bpc.cm.items.routing.pareto.population.Individual;

public class PopSizeTest {

	SolomonProblem problemName;
	int customerNumber;

	@Test
	public void testProblem() {

//		for (int t = 10; t <= 40; t+=5){ // for R & RC
		for (int t = 70; t <= 100; t += 5) { // for C
			problemName = SolomonProblem.C101;
			customerNumber = t;
			Matrix m = MatrixTest.testSolomon(problemName, customerNumber);

			int routesNumber = 0;
			int totalCost = 0;

			if (problemName.equals("C201")) {
				if (customerNumber == 70) {
					routesNumber = 3;
					totalCost = 4993;
				} else if (customerNumber == 75) {
					routesNumber = 3;
					totalCost = 5092;
				} else if (customerNumber == 80) {
					routesNumber = 3;
					totalCost = 5223;
				} else if (customerNumber == 85) {
					routesNumber = 3;
					totalCost = 5250;
				} else if (customerNumber == 90) {
					routesNumber = 3;
					totalCost = 5452;
				} else if (customerNumber == 95) {
					routesNumber = 3;
					totalCost = 5742;
				} else {
					routesNumber = 3;
					totalCost = 5891;
				}
			} else if (problemName.equals("R201")) {
				if (customerNumber == 10) {
					routesNumber = 1;
					totalCost = 2536;
				} else if (customerNumber == 15) {
					routesNumber = 1;
					totalCost = 3529;
				} else if (customerNumber == 20) {
					routesNumber = 2;
					totalCost = 4093;
				} else if (customerNumber == 25) {
					routesNumber = 2;
					totalCost = 5227;
				} else if (customerNumber == 30) {
					routesNumber = 2;
					totalCost = 5737;
				} else if (customerNumber == 35) {
					routesNumber = 2;
					totalCost = 6632;
				} else {
					routesNumber = 2;
					totalCost = 7926;
				}
			} else {
				if (customerNumber == 10) {
					routesNumber = 1;
					totalCost = 1941;
				} else if (customerNumber == 15) {
					routesNumber = 1;
					totalCost = 3349;
				} else if (customerNumber == 20) {
					routesNumber = 2;
					totalCost = 3982;
				} else if (customerNumber == 25) {
					routesNumber = 2;
					totalCost = 4313;
				} else if (customerNumber == 30) {
					routesNumber = 2;
					totalCost = 6796;
				} else if (customerNumber == 35) {
					routesNumber = 2;
					totalCost = 7587;
				} else {
					routesNumber = 2;
					totalCost = 9337;
				}
			}

			System.out.println(problemName + " - " + customerNumber + "\npop-size, Optimal (%)");
			for (int i = 1; i <= 40; i++) {
				int optimalFound = 0;
				int totalTime = 0;
				for (int j = 0; j < 10; j++) {
					Pareto p = new Pareto(m);
					Pareto.POPULATION_SIZE = i * 10;
					Pareto.EUCLIDEAN_RADIUS = 30;
					long time = System.currentTimeMillis();
					Individual result = p.computeResult();
					// System.out.println(result);
					totalTime += System.currentTimeMillis() - time;

					if (result.getRoutesNumber() == routesNumber && result.getTotalCost() < totalCost * 1.05) {
						optimalFound++;
					}
				}
				System.out.println((i * 10) + "," + optimalFound * 10);

			}
		}

	}

}