package simulator;

import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	public static void main(String[] args){
		try {
			FIS fis = FIS.load("cogumelos.fcl", true);
			
			Simulator simulator = new Simulator();
			FunctionBlock fb = fis.getFunctionBlock("angleAction");
		
		
			DataSource source = new DataSource("mushroom.arff");
			Instances data = source.getDataSet();	
			data.setClassIndex(data.numAttributes() - 1);	
			
			//Generate model
			J48 classifier = new J48();
			classifier.buildClassifier(data);
			
		
			while(true) {
				if(simulator.getMushroomAttributes()!=null) {
					NewInstances ni = new NewInstances(data);
					ni.addInstance(simulator.getMushroomAttributes());
					double predict = classifier.classifyInstance(data.lastInstance());
					fb.setVariable("classification", predict);
				}
				
				double r = simulator.getDistanceR(), l= simulator.getDistanceL(), c = simulator.getDistanceC();
				fb.setVariable("distanceRIGHT", r);
				fb.setVariable("distanceLEFT", l);
				fb.setVariable("distanceCENTER", c);
				fb.evaluate();
				fb.getVariable("angle").defuzzify();
				fb.getVariable("action").defuzzify();
				if(fb.getVariable("action").getValue()==0.0 && Math.min(l,Math.min(c,r))<=1.0) simulator.setAction(Action.DESTROY);
				else if(fb.getVariable("action").getValue()==1.0 && Math.min(l,Math.min(c,r))<=1.0) simulator.setAction(Action.PICK_UP);
				else simulator.setAction(Action.NO_ACTION);
				simulator.setRobotAngle(fb.getVariable("angle").getValue());
				simulator.step();
			}
		
		}catch (Exception e) {
			e.printStackTrace();
		}
	}

}
