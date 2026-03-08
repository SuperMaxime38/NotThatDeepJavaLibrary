package ntdjl;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class MainTest {
	
	public static void main(String[] args) {
		
		String filepath = "C:/Users/maxime/Documents/model.ntdjl";
		
		NN model = new NN();
		
//		try {
//			model.load(filepath);
//			System.out.println("Model loaded");
//			System.out.println("Prediction: " + model.predictList(new SimpleMatrix(new double[][] {{235, 69}}).transpose()));
//			
//			return;
//		} catch(IOException e) {
//			e.printStackTrace();
//		} catch (ClassNotFoundException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		int[] n = {2, 3, 3, 1};
		
		model.addLayer(new Layer(n[0], n[1], ActivationFunction.SIGMOID));
		model.addLayer(new Layer(n[1], n[2], ActivationFunction.SIGMOID));
		model.addLayer(new Layer(n[2], n[3], ActivationFunction.SIGMOID));
		
		
		double[][] x = {
			{150, 70},
			{254, 73},
			{312, 68},
			{120, 60},
			{154, 61},
			{212, 65},
			{216, 67},
			{145, 67},
			{184, 64},
			{130, 69}
		};
		
		
		//Random test don't mind
//		DataHolder holder = new DataHolder();
//		Random rdm = new Random();
//		for(int i = 0; i < 10; i++) {
//			ArrayList<Double> ls = new ArrayList<Double>();
//			ls.add((double) rdm.nextInt(100, 200));
//			ls.add((double) rdm.nextInt(60, 80));
//			holder.addData(ls);
//		}
//		
//		SimpleMatrix A0 = holder.convertToMatrix();
		
		// So the previous code piece might be used in my plugin in order to collect datas from environment
		
		SimpleMatrix A0 = new SimpleMatrix(x);
		A0 = A0.transpose();

		double[] y = {
			0,
			1,
			1,
			0,
			0,
			1,
			1,
			0,
			1,
			0
		};

		SimpleMatrix Y = new SimpleMatrix(y);
		int m = 10;
		
		Y.reshape(n[3], m);
		
		model.train(A0, Y, 10000, 0.1);
		
//		double[][] test = {
//				{235, 69}
//		};
//		
//		SimpleMatrix testMat = new SimpleMatrix(test);
//		testMat = testMat.transpose();
//		
//		//Pair out = model.feed_forward(testMat);
//		List<Float> res = model.predictList(testMat);
//		System.out.println("Final res: " + res);
//		System.out.println("In matrxi val: " + model.predict(testMat));
//
//		
//		try {
//			model.save(filepath);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
	}
}
