package pareto;

public enum SolomonProblem {
	C101("C101"), C102("C102"), C103("C103"), C104("C104"), C105("C105"), C106("C106"), C107("C107"), C108("C108"),
			C109("C109"),

	C201("C201"), C202("C202"), C203("C203"), C204("C204"), C205("C205"), C206("C206"), C207("C207"), C208("C208"),

	R101("R101"), R102("R102"), R103("R103"), R104("R104"), R105("R105"), R106("R106"), R107("R107"), R108("R108"),
			R109("R109"), R110("R110"), R111("R111"), R112("R112"),

	R201("R201"), R202("R202"), R203("R203"), R204("R204"), R205("R205"), R206("R206"), R207("R207"), R208("R208"),
			R209("R209"), R210("R210"), R211("R211"),

	RC101("RC101"), RC102("RC102"), RC103("RC103"), RC104("RC104"), RC105("RC105"), RC106("RC106"), RC107("RC107"),
			RC108("RC108"),

	RC201("RC201"), RC202("RC202"), RC203("RC203"), RC204("RC204"), RC205("RC205"), RC206("RC206"), RC207("RC207"),
			RC208("RC208");

	String name;

	SolomonProblem(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public String toString() {
		return name;
	}
}
