package simulator;
import java.util.Scanner;

import javax.swing.ImageIcon;

public class Mushroom {
	private String odor;
	private String sporePrintColor;
	private String capShape;
	private String classification;

	public Mushroom(String odor, String sporePrintColor, String capShape, String classification) {
		super();
		this.odor = odor;
		this.sporePrintColor = sporePrintColor;
		this.capShape = capShape;
		this.classification = classification;
	}

	public Mushroom(Scanner s) {
		// pungent,black,convex,poisonous
		String[] line = s.nextLine().split(",");
		this.odor = line[0];
		this.sporePrintColor = line[1];
		this.capShape = line[2];
		this.classification = line[3];
	}

	public String getOdor() {
		return odor;
	}

	public String getSporePrintColor() {
		return sporePrintColor;
	}

	public String getCapShape() {
		return capShape;
	}

	public String getClassification() {
		return classification;
	}

	@Override
	public String toString() {
		return "Mushroom [odor=" + odor + ", sporePrintColor=" + sporePrintColor + ", capShape=" + capShape
				+ ", classification=" + classification + "]";
	}

	public ImageIcon getImage() {
		if (classification.equals("poisonous"))
			return new ImageIcon("poisonous.jpeg"); //alteração
		else
			return new ImageIcon("eatable.jpg");

	}

	public String[] getAttributes() {
		String[] values = new String[3];
		values[0] = odor;
		values[1] = sporePrintColor;
		values[2] = capShape;
		return values;
	}
}
