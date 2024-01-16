package pareto;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.junit.Test;

import ru.bpc.cm.items.routing.Matrix;
import ru.bpc.cm.items.routing.SolutionRoutes;
import ru.bpc.cm.items.routing.pareto.Pareto;
import ru.bpc.cm.items.routing.pareto.problem.Problem;
  
  
public class BrokenProblemTest {
	
	// 28/03/2015
	public static Matrix test5() {	
		Matrix m = new Matrix(20);
		m.ENC = new int[]{0, 22, 14, 8, 9, 19, 24, 18, 17, 23, 11, 26, 16, 28, 7, 13, 10, 15, 27, 12};
		m.ATM = new String[]{"1", "202524", "202576", "202580", "202640", "202670", "202682", "202688", "202722", "202748", "202784", "202790", "202800", "210004", "300284", "300454", "451013", "451054", "451104", "451106"};
		m.distanceCoeffs = new int[][]{{0, 10, 19, 16, 8, 15, 8, 13, 16, 10, 19, 8, 8, 14, 10, 16, 19, 6, 8, 6}, {17, 0, 17, 11, 22, 8, 12, 5, 18, 26, 17, 6, 9, 11, 21, 18, 13, 21, 10, 12}, {20, 12, 0, 13, 25, 15, 14, 14, 21, 37, 15, 9, 23, 21, 14, 20, 13, 23, 22, 14}, {5, 4, 12, 0, 6, 11, 12, 20, 12, 12, 3, 15, 25, 14, 7, 6, 19, 22, 12, 8}, {21, 20, 11, 25, 0, 9, 11, 10, 17, 29, 21, 10, 7, 8, 14, 6, 11, 20, 11, 7}, {18, 18, 12, 23, 10, 0, 13, 2, 20, 27, 19, 4, 13, 13, 10, 7, 12, 16, 10, 7}, {15, 13, 5, 20, 3, 4, 0, 7, 14, 22, 15, 4, 11, 9, 13, 18, 16, 5, 25, 21}, {13, 6, 10, 5, 13, 11, 18, 0, 8, 9, 3, 14, 14, 11, 14, 17, 12, 6, 20, 20}, {12, 5, 9, 10, 11, 9, 16, 12, 0, 14, 4, 11, 21, 14, 10, 12, 13, 13, 15, 14}, {4, 11, 10, 16, 2, 1, 12, 12, 20, 0, 11, 7, 17, 9, 22, 25, 27, 7, 32, 37}, {20, 6, 11, 17, 20, 18, 25, 15, 11, 9, 0, 20, 21, 17, 9, 10, 16, 11, 10, 12}, {5, 10, 10, 4, 15, 4, 10, 10, 18, 10, 8, 0, 9, 7, 13, 10, 11, 11, 13, 12}, {5, 9, 8, 6, 14, 6, 10, 10, 18, 9, 5, 15, 0, 14, 6, 11, 20, 11, 2, 7}, {18, 16, 12, 23, 10, 11, 19, 27, 19, 3, 13, 13, 20, 0, 21, 13, 17, 18, 14, 9}, {14, 12, 13, 13, 12, 20, 13, 10, 15, 13, 5, 28, 37, 11, 0, 28, 26, 21, 9, 15}, {20, 9, 20, 20, 27, 12, 12, 23, 24, 16, 13, 19, 15, 3, 22, 0, 21, 13, 2, 5}, {11, 7, 10, 11, 19, 11, 11, 14, 15, 12, 8, 9, 16, 11, 3, 4, 0, 15, 12, 8}, {20, 7, 8, 7, 16, 23, 15, 12, 12, 9, 20, 19, 6, 14, 8, 21, 19, 0, 10, 25}, {9, 11, 12, 12, 24, 22, 12, 4, 9, 20, 15, 7, 14, 8, 14, 19, 10, 19, 0, 8}, {10, 11, 7, 18, 14, 12, 3, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 0}};
		m.timeCoeffs = new int[][]{{0, 23, 29, 28, 18, 28, 19, 26, 27, 20, 36, 16, 15, 25, 20, 23, 30, 15, 18, 130}, {28, 0, 26, 20, 39, 19, 18, 9, 21, 33, 28, 10, 12, 17, 35, 30, 26, 33, 21, 20}, {30, 21, 0, 21, 35, 24, 19, 23, 22, 35, 28, 15, 29, 28, 28, 32, 23, 34, 32, 24}, {16, 10, 20, 0, 16, 22, 18, 22, 14, 21, 8, 20, 29, 21, 16, 13, 27, 33, 21, 17}, {32, 30, 23, 43, 0, 19, 19, 13, 20, 36, 31, 13, 9, 13, 24, 15, 19, 29, 18, 12}, {27, 25, 19, 38, 18, 0, 17, 5, 21, 33, 27, 5, 17, 20, 22, 16, 16, 25, 18, 16}, {23, 22, 13, 34, 11, 10, 0, 9, 16, 28, 22, 6, 15, 16, 25, 31, 28, 16, 34, 32}, {23, 17, 19, 14, 22, 17, 22, 0, 12, 16, 9, 20, 25, 17, 31, 29, 19, 16, 32, 30}, {21, 19, 16, 27, 19, 14, 21, 17, 0, 26, 13, 19, 27, 23, 20, 22, 18, 21, 25, 23}, {11, 20, 16, 31, 8, 4, 14, 14, 26, 0, 19, 12, 21, 14, 42, 47, 33, 17, 49, 46}, {39, 18, 26, 35, 37, 33, 33, 23, 20, 21, 0, 32, 36, 29, 19, 25, 24, 23, 25, 26}, {13, 22, 23, 12, 33, 11, 18, 17, 28, 21, 16, 0, 20, 15, 25, 18, 13, 17, 21, 18}, {10, 15, 12, 10, 26, 13, 10, 10, 22, 15, 9, 17, 0, 17, 12, 15, 25, 16, 4, 10}, {24, 22, 17, 32, 17, 15, 20, 32, 23, 5, 16, 19, 25, 0, 24, 19, 21, 25, 17, 15}, {18, 15, 23, 19, 13, 20, 18, 14, 18, 15, 10, 36, 35, 21, 0, 35, 36, 29, 17, 28}, {27, 19, 30, 25, 32, 17, 22, 30, 30, 24, 28, 32, 27, 9, 35, 0, 33, 24, 12, 15}, {20, 17, 23, 18, 23, 15, 21, 21, 27, 19, 14, 11, 21, 16, 6, 6, 0, 20, 18, 13}, {28, 14, 11, 9, 17, 29, 19, 16, 20, 15, 27, 29, 11, 22, 14, 28, 26, 0, 18, 35}, {16, 15, 17, 15, 30, 26, 17, 7, 16, 27, 27, 13, 21, 14, 26, 26, 18, 33, 0, 16}, {15, 16, 14, 29, 24, 18, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
		m.addTimeWindow(0, 0, 10000, false);
		m.addTimeWindow(1, 600, 1380, false);
		m.addTimeWindow(2, 600, 1380, false);
		m.addTimeWindow(3, 600, 1380, false);
		m.addTimeWindow(4, 600, 1380, false);
		m.addTimeWindow(5, 600, 1380, false);
		m.addTimeWindow(6, 600, 1380, false);
		m.addTimeWindow(7, 600, 1380, false);
		m.addTimeWindow(8, 600, 1380, false);
		m.addTimeWindow(9, 600, 1380, false);
		m.addTimeWindow(10, 600, 1380, false);
		m.addTimeWindow(11, 600, 1380, false);
		m.addTimeWindow(12, 600, 1380, false);
		m.addTimeWindow(13, 600, 1380, false);
		m.addTimeWindow(14, 600, 1380, false);
		m.addTimeWindow(15, 600, 1380, false);
		m.addTimeWindow(16, 600, 1380, false);
		m.addTimeWindow(17, 600, 1380, false);
		m.addTimeWindow(18, 600, 1380, false);
		m.addTimeWindow(19, 60, 100, false);
		m.addRiderTimeWindow(60, 1380);
		m.amountOfMoney = new int[]{0, 17388850, 0, 11340000, 12500000, 12750000, 10700000, 10690000, 12190000, 12750000, 10640000, 12750000, 10660000, 6160000, 20321080, 17135990, 18528760, 24149390, 16388850, 17560000};
		m.serviceTime = new int[]{15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
		m.MaxMoney = 150000000;
		m.amountOfCassettes = new int[]{0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200};
		m.VolumeOneCar = 150000000;
		m.FixPrice = 3500.0;
		m.LengthPrice = 50.0;
		m.MaxATMInWay = 13;
		m.MaxTime = 0;
		m.MaxLength = 0;
		m.depot = "1";
		m.maxCars = 5;
		m.AtmPrice = new double[]{3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 3500.0};
		m.currCode = 810;
		m.windowMode = 0;
		return m;
	/*
	 ~~~~~~~~~~~~~~~Riders~~~~~~~~~~~~~~~~~~~
	   Rider:
		 atmList = [2, 16, 15, 13, 9, 5, 11, 3, 14]
		 cost = [4465.0, 9595.0, 14940.0, 20450.0, 26125.0, 31865.0, 37820.0, 43990.0, 50525.0]
		 time = [89, 127, 621, 645, 665, 684, 704, 731, 762]
		 timewait = [0, 473, 0, 0, 0, 0, 0, 0, 0]
		 length = [19, 13, 4, 3, 3, 1, 4, 4, 7]
		 sumWaitTime = 473
		 sumTime = 5636
		 Money = 111735830
		 Cassettes = 1800
		 SumLength = 0
		 WaitInDepoTime = 511
		 WayNum = 0
	   Rider:
		 atmList = [4, 19, 6, 17, 12, 18, 7, 8, 10, 1]
		 cost = [3915.0, 8195.0, 12640.0, 17350.0, 22375.0, 27515.0, 32870.0, 38640.0, 44625.0, 50925.0]
		 time = [78, 105, 127, 631, 657, 676, 698, 725, 753, 786]
		 timewait = [0, 0, 473, 0, 0, 0, 0, 0, 0, 0]
		 length = [8, 7, 3, 5, 6, 2, 4, 8, 4, 6]
		 sumWaitTime = 473
		 sumTime = 5859
		 Money = 142867090
		 Cassettes = 2000
		 SumLength = 0
		 WaitInDepoTime = 522
		 WayNum = 0
	 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 Rider Number: 1Route Number1 Route time: 89 Route cost: 4465.0 Route length: 19 Rider money: 111735830 Route: [14, 10, 13, 28, 23, 19, 26, 8, 7]
	 Rider Number: 2Route Number1 Route time: 78 Route cost: 3915.0 Route length: 8 Rider money: 142867090 Route: [9, 12, 24, 15, 16, 27, 18, 17, 11, 22]
	*/
	}
    	
	@Test
	public void testProblem() {
		Matrix m = test5();
		
    	Pareto solver = new Pareto(m);
    	Pareto.SHOW_DEBUG = true;
        
    	SolutionRoutes result = solver.computeResult();
    	
    	System.out.println(result.getStatus());
    	
    	for (ArrayList<Integer> route : result.getRoutes()) {
    		System.out.print("[");
    		for (Integer pid : route) {
    			System.out.print(pid + " ");
    		}
    		System.out.println("]");
    	}
    	
    	for (ArrayList<Integer> route : result.getRoutes()) {
    		int order = 1;
    		int pointTime = m.getTimeWindows().get(0).StartWork; // start time of depot
    		pointTime -= m.serviceTime[0]; // if not serving depot in the beginning
    		int prevCust = 0;
    		for (Integer pid : route) {
    			pointTime = Math.max(pointTime + m.timeCoeffs[prevCust][pid] + m.serviceTime[prevCust], m.getTimeWindows().get(pid).StartWork);
    			System.out.println(pid + " " + order + " "  +pointTime);
    			prevCust = pid;
    			order++;
    		}
    	}
    	System.out.println("coef  ->  matrix    problem");
    	Problem problem = Problem.getInstance(); 
    	for (int i = 1; i < m.timeCoeffs.length; i++) {
    			System.out.println(i + ": " +m.timeCoeffs[0][i] + "  ->  (" + m.getTimeWindow(i).StartWork +
    					", " + m.getTimeWindow(i).EndWork + ")    (" + problem.getCustomer(i-1).getReadyTime() +
    					", " + problem.getCustomer(i-1).getDueDate() + ")");
    	}

	}
    	 	
}  