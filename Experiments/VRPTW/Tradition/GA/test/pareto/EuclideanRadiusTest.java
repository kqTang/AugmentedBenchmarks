package pareto;

import org.junit.Test;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.pareto.Pareto;

public class EuclideanRadiusTest {

	SolomonProblem problemName = SolomonProblem.C101;

	@Test
	public void testProblem() {
		Matrix m = MatrixTest.testSolomon(problemName, 100);
		System.out.println(problemName);
		for (int r = 20; r <= 800; r += 20) {
			System.out.println("Radius = " + r);
			for (int i = 0; i < 5; i++) {
				Pareto p = new Pareto(m);
				Pareto.POPULATION_SIZE = 100;
				Pareto.EUCLIDEAN_RADIUS = r;
				Pareto.SHOW_GEN = true;

				p.computeResult();
			}

		}

	}

}