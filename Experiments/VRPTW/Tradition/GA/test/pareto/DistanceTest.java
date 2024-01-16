package pareto;

import org.junit.Test;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.pareto.Pareto;
import ru.bpc.cm.items.routing.pareto.problem.Problem;

public class DistanceTest {

	SolomonProblem problemName = SolomonProblem.C101;

	@Test
	public void testProblem() {
		Matrix m = MatrixTest.testSolomon(problemName, 100);
		new Pareto(m);
		int[][] distanceCoeffs = Problem.getInstance().distanceCoeffs;
		int counter = 0, total = 0, max = 0;
		for (int i = 0; i < distanceCoeffs.length; i++) {
			for (int j = 0; j < i; j++) {
				// System.out.print(distanceCoeffs[i][j] + " ");
				max = max > distanceCoeffs[i][j] ? max : distanceCoeffs[i][j];
				total += distanceCoeffs[i][j];
				counter++;
			}
			// System.out.println();
		}
		System.out.println("Max = " + max);
		System.out.println("Avg = " + (total / counter));

	}

}