package simulator;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Point;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.Scanner;
import javax.swing.JComponent;
import javax.swing.JFrame;

/**
 * 
 * @author Sancho Oliveira
 * @version 5
 *
 */
public class Simulator {
	private int points = 0;
	private static final double MIN_DISTANCE = 30;
	private static final double SENSING_DISTANCE = 50;
	JFrame f = new JFrame("Robot simulator");
	private Arena arena;
	private double angle;
	private Action action = Action.NO_ACTION;

	private Point robotPosition = new Point(50, 250);
	private int robotSpeed = 5;
	private double robotDirection = Math.PI / 6;
	private int simulationSpeed = 100;

	private Point mushroom;
	private LinkedList<Mushroom> mushrooms = new LinkedList<Mushroom>();

	private double distanceC;
	private double obstAngleC;
	private double distanceL;
	private double distanceR;
	private double obstAngleR;
	private double obstAngleL;
	private int index;

	public Simulator() {
		arena = new Arena();
		f.add(arena);
		f.setSize(500, 500);
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setVisible(true);
		loadMushrooms();
		// System.out.println(mushrooms);
		addMushroom();

	}

	private void loadMushrooms() {
		try {
			Scanner s = new Scanner(new File("mushroom.arff"));
			for (int i = 0; i < 8; i++)
				s.nextLine();
			while (s.hasNext()) {
				mushrooms.add(new Mushroom(s));
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private void addMushroom() {
		int x = (int) (Math.random() * f.getContentPane().getWidth());
		int y = (int) (Math.random() * f.getContentPane().getHeight());
		index = (int) (Math.random() * mushrooms.size());
		mushroom = new Point(x, y);
	}


	/**
	 * Sets the new angle for the wheels of the robot.
	 * 
	 * @param angle the value of the new angle in degrees
	 */
	public void setRobotAngle(double angle) {
		this.angle = Math.toRadians(angle);
	}

	/**
	 * Setter for the action for the robot to perform in the next time step.
	 * Possible actions are: 
	 *  - DESTROY: used to instruct the robot to destroy the mushroom, 
	 *  - NO_ACTION: the robot does not perform any action, 
	 *  - PICK_UP: used to instruct the robot to pick up the mushroom.
	 * 
	 * @param action to be performed.
	 */
	public void setAction(Action action) {
		this.action = action;
	}

	/**
	 * Getter for the robot speed.
	 * 
	 * @return the current speed of the robot
	 */
	public int getRobotSpeed() {
		return robotSpeed;
	}

	/**
	 * Getter for the attributes of the observed mushroom.
	 * 
	 * @return an array with the mushroom's attributes
	 */
	public String[] getMushroomAttributes() {
		if (distanceC > SENSING_DISTANCE)
			return null;
		return mushrooms.get(index).getAttributes();

	}
	

	/**
	 * Sets the new robot speed.
	 * 
	 * @param robotSpeed the new value for the robot speed.
	 */
	public void setRobotSpeed(int robotSpeed) {
		this.robotSpeed = robotSpeed;
	}

	/**
	 * Getter for the simulation speed
	 * 
	 * @return time for a simulation step in milliseconds.
	 */
	public int getSimulationSpeed() {
		return simulationSpeed;
	}

	/**
	 * Setter for the time needed for a simulation step.
	 * 
	 * @param simulationSpeed time in milliseconds.
	 */
	public void setSimulationSpeed(int simulationSpeed) {
		this.simulationSpeed = simulationSpeed;
	}

	/**
	 * Moves simulation one time step forward and shows the new state of the
	 * environment in GUI. Uses the current angle of the wheels to move the robot.
	 * After the new position of the robot has been computes the sensor values are
	 * updated.
	 */
	public void step() {
		if (action == Action.NO_ACTION) {
			robotDirection += angle;
			robotDirection = robotDirection % (2 * Math.PI);
			robotPosition.x += (robotSpeed * Math.cos(robotDirection));
			robotPosition.y += (robotSpeed * Math.sin(robotDirection));
			if (robotPosition.x > f.getContentPane().getWidth())
				robotPosition.x = 0;
			if (robotPosition.x < 0)
				robotPosition.x = f.getContentPane().getWidth();
			if (robotPosition.y > f.getContentPane().getHeight())
				robotPosition.y = 0;
			if (robotPosition.y < 0)
				robotPosition.y = f.getContentPane().getHeight();
		} else if (Math.min(distanceL, Math.min(distanceC, distanceR)) < MIN_DISTANCE) {
			if (action == Action.PICK_UP) {
				if (mushrooms.get(index).getClassification().equals("poisonous")) {
					points -= 10;
				} else {
					points += 1;
				}
			} else if (action == Action.DESTROY) {
				if (mushrooms.get(index).getClassification().equals("poisonous")) {
					points += 1;
				} else {
					points -= 5;
				}
			}
			f.repaint();
			try {
				Thread.sleep(500);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			addMushroom();

		}
		updateSensorValues();
		f.repaint();

		try {
			Thread.sleep(simulationSpeed);
		} catch (InterruptedException e) {
		}

	}

	private void updateSensorValues() {
		distanceL = Double.MAX_VALUE;
		distanceC = Double.MAX_VALUE;
		distanceR = Double.MAX_VALUE;

		double dist = robotPosition.distance(mushroom);
		double ang = Math.atan2(mushroom.y - robotPosition.y, mushroom.x - robotPosition.x);
		double angToRobot = (ang - robotDirection) % (2 * Math.PI);
		if (angToRobot > Math.PI)
			angToRobot -= 2 * Math.PI;
		if (angToRobot < -Math.PI)
			angToRobot += 2 * Math.PI;
		if (Math.abs(angToRobot) <= Math.PI / 6) {
			distanceC = dist;
			obstAngleC = ang;
		} else if (angToRobot > Math.PI / 6 && angToRobot < Math.PI / 2) {
			distanceR = dist;
			obstAngleR = ang;
		} else if (angToRobot < -Math.PI / 6 && angToRobot > -Math.PI / 2) {
			distanceL = dist;
			obstAngleL = ang;
		}
	}

	/**
	 * Getter for the distance of the closest obstacle detected by the central
	 * sensor
	 * 
	 * @return distance in meters of the closest obstacle inside the opening angle
	 *         (60º) of the central sensor.
	 */
	public double getDistanceC() {
		return Math.min(10, distanceC / 30);
	}

	/**
	 * Getter for the distance of the closest obstacle detected by the left sensor
	 * 
	 * @return distance in meters of the closest obstacle inside the opening angle
	 *         (60º) of the left sensor.
	 */
	public double getDistanceL() {
		return Math.min(10, distanceL / 30);
	}

	/**
	 * Getter for the distance of the closest obstacle detected by the right sensor
	 * 
	 * @return distance in meters of the closest obstacle inside the opening angle
	 *         (60º) of the right sensor.
	 */
	public double getDistanceR() {
		return Math.min(10, distanceR / 30);
	}

	private class Arena extends JComponent {
		private static final int ROBOT_RADIUS = 10;
		private static final int OBST_RADIUS = 50;

		@Override
		public void paint(Graphics g) { 
			super.paint(g);
			g.setFont(new Font("Courier New", 1, 17));
			g.drawString("Points: " + points, 30, 30);
			
			if(action == Action.NO_ACTION) {
				g.drawString("Looking for Mushroom", 30, 60);
			}else{
			if(action == Action.DESTROY) {
				g.drawString("Destroyed!", 30, 60);
				
			}else{
			if(action == Action.PICK_UP) {
				g.drawString("Picked Up", 30, 60);
			}}}
			
			
			g.setColor(Color.black);
			g.fillOval(mushroom.x - OBST_RADIUS / 2, mushroom.y - OBST_RADIUS / 2, OBST_RADIUS, OBST_RADIUS);
			g.drawImage(mushrooms.get(index).getImage().getImage(), mushroom.x - OBST_RADIUS / 2,
					mushroom.y - OBST_RADIUS / 2, OBST_RADIUS, OBST_RADIUS, null);

			g.drawOval(robotPosition.x - ROBOT_RADIUS / 2, robotPosition.y - ROBOT_RADIUS / 2, ROBOT_RADIUS,ROBOT_RADIUS);

			int x = robotPosition.x; // + ROBOT_RADIUS / 2 ;
			int y = robotPosition.y;// + ROBOT_RADIUS / 2
			
			
			g.setColor(Color.orange);
			g.drawLine(x, y, (int) (x + 10 * Math.cos(robotDirection)), (int) (y + 10 * Math.sin(robotDirection)));
			g.setColor(Color.black);
			g.drawLine(x, y, (int) (x + 10 * Math.cos(robotDirection+30)), (int) (y + 10 * Math.sin(robotDirection+30)));
			g.drawLine(x, y, (int) (x + 10 * Math.cos(robotDirection-30)), (int) (y + 10 * Math.sin(robotDirection-30)));

			if (distanceC < Double.MAX_VALUE) {
				g.setColor(Color.red);
				g.drawLine(x, y, (int) (x + distanceC * Math.cos(obstAngleC)),
						(int) (y + distanceC * Math.sin(obstAngleC)));
			}
			if (distanceL < Double.MAX_VALUE) {
				g.setColor(Color.green);
				g.drawLine(x, y, (int) (x + distanceL * Math.cos(obstAngleL)),
						(int) (y + distanceL * Math.sin(obstAngleL)));
			}

			if (distanceR < Double.MAX_VALUE) {
				g.setColor(Color.blue);
				g.drawLine(x, y, (int) (x + distanceR * Math.cos(obstAngleR)),
						(int) (y + distanceR * Math.sin(obstAngleR)));
			}
		}
	}

}
