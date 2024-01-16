package ru.bpc.cm.items.routing.pareto.common;

import java.util.Random;

public class ArrayShuffler {
	// Implementing Fisher-Yates shuffle
	public static void shuffle(int[] ar) {
		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}
}
