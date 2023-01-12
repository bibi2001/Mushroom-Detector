package simulator;


import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NewInstances {
	Instances database = null;

	public NewInstances(Instances database) {
		database.clear();
		this.database = database;
	}

	public void addInstance(String[] attributes) {
		int nrAttr = database.numAttributes();
		if (nrAttr == attributes.length+1) {
			double[] instancesValue = new double[nrAttr];
			for (int i = 0; i < nrAttr-1; i++) {
				instancesValue[i] = database.attribute(i).indexOfValue(attributes[i]);
			}
			database.add(new DenseInstance(1.0, instancesValue));
		} else {
			System.out.println("Incorrect number of attributes");
		}
	}

	public Instances getDataset() {
		return database;
	}


}
