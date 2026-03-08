package ntdjl;

import java.io.Serializable;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class Layer implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private SimpleMatrix weights;
	private SimpleMatrix bias;
	private ActivationFunction activ;
	
	// Pour stocker temporairement l’entrée (z) et la sortie (a)
    public SimpleMatrix lastZ, lastA;
	
	public Layer(int inputSize, int outputSize, ActivationFunction fn) {
		Random rand = new Random();
        weights = SimpleMatrix.random_DDRM(outputSize, inputSize, -1.0, 1.0, rand);
        bias = SimpleMatrix.random_DDRM(outputSize, 1, -1.0, 1.0, rand);
		this.activ = fn;
		//System.out.println("Layer : " + weights.toString() + "\n" + bias.toString());
	}
	
	public SimpleMatrix forward(SimpleMatrix input) {
	    SimpleMatrix z = weights.mult(input);

	    // Broadcasting du biais
//	    SimpleMatrix biasMatrix = new SimpleMatrix(bias.getNumRows(), input.getNumCols());
//	    for (int i = 0; i < input.getNumCols(); i++) {
//	        biasMatrix.insertIntoThis(0, i, bias);
//	    }
	    SimpleMatrix ones = new SimpleMatrix(1, z.getNumCols());
	    ones.fill(1.0);;
	    
	    SimpleMatrix biasMatrix = bias.mult(ones);
	    //System.out.println("bias: " + bias.toString() + "\nbiasMatrix: " + biasMatrix.toString());

	    
	    this.lastZ = z.plus(biasMatrix);
	   // System.out.println("Output: " + this.lastZ.toString());
	    
	    applyActivationFunction();
	    
        return this.lastA;
	}
	
	// Applique la rétropropagation sur ce layer
    public SimpleMatrix backward(SimpleMatrix dA, SimpleMatrix A_prev, double learningRate) {
        SimpleMatrix dZ = dA.elementMult(MathsAreGood.sigmoidDerivative(this.lastA));
        SimpleMatrix dW = dZ.mult(A_prev.transpose()).divide(A_prev.getNumCols());
        SimpleMatrix dB = MathsAreGood.meanRows(dZ);
        SimpleMatrix dA_prev = weights.transpose().mult(dZ);

        // Met à jour les poids
        weights = weights.minus(dW.scale(learningRate));
        bias = bias.minus(dB.scale(learningRate));

        return dA_prev;
    }
    
    public void applyActivationFunction() {
		switch (this.activ) {
		case SIGMOID:
			this.lastA = MathsAreGood.sigmoid(this.lastZ);
			break;
		case RELU:
			this.lastA = MathsAreGood.relu(this.lastZ);
			break;
		case GELU:
			this.lastA = MathsAreGood.gelu(this.lastZ);
			break;
		case ELU:
			this.lastA = MathsAreGood.elu(this.lastZ);
			break;
		default:
			this.lastA = this.lastZ;
			break; // DEFAULT IS ALWAYS_ACTIVE
		}
	}
    
    public Layer clone() {
        Layer copy = new Layer(0, 0, this.activ); // tailles fictives pour init
        copy.weights = this.weights.copy();
        copy.bias = this.bias.copy();
        copy.activ = this.activ; // enum, donc partagé sans souci
        return copy;
    }
    
    public void mutate(double rate) {
        Random rand = new Random();

        for (int i = 0; i < weights.getNumRows(); i++) {
            for (int j = 0; j < weights.getNumCols(); j++) {
                double mutation = rate * (rand.nextGaussian()); // bruit gaussien
                weights.set(i, j, weights.get(i, j) + mutation);
            }
        }

        for (int i = 0; i < bias.getNumRows(); i++) {
            double mutation = rate * (rand.nextGaussian());
            bias.set(i, 0, bias.get(i, 0) + mutation);
        }
    }
    
    public double getWeight(int i, int j) {
    	return this.weights.get(i, j);
    }
    
    public double getBias(int i) {
		return this.bias.get(i, 0);
	}

	
	public SimpleMatrix getWeights() {
		return this.weights;
	}
	
	public SimpleMatrix getBias() {
		return this.bias;
	}
	
	public ActivationFunction getActiv() {
		return this.activ;
	}

	public void setWeight(int i, int j, double value) {
		weights.set(i, j, value);
	}
	
	public void setBias(int i, double value) {
		weights.set(i, 0, value);
	}
	
	public void setWeights(SimpleMatrix weights) {
		this.weights = weights;
	}
	
	public void setBias(SimpleMatrix bias) {
		this.bias = bias;
	}
	
	public void setActiv(ActivationFunction activ) {
		this.activ = activ;
	}
}
