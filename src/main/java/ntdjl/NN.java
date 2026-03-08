package ntdjl;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;
import ntdjl.utils.Pair;

public class NN {
	private ArrayList<Layer> layers;
	
	public NN() {
		layers = new ArrayList<Layer>();
	}
	
	public void addLayer(Layer layer) {
		layers.add(layer);
	}
	
	public ArrayList<Layer> getLayers() {
		return layers;
	}
	
	public void removeLayer(Layer layer) {
		layers.remove(layer);
	}
	
	public Pair feed_forward(SimpleMatrix input) {
		SimpleMatrix output = input;
		ArrayList<SimpleMatrix> layers_outputs = new ArrayList<SimpleMatrix>();
	    for (Layer layer : layers) {
	        output = layer.forward(output);
	        layers_outputs.add(output);
	    }
	    
	    return new Pair(output, layers_outputs);
	}
	
	public double cost(SimpleMatrix yHat, SimpleMatrix y) {
	    // Différence entre la prédiction et la vérité
	    SimpleMatrix diff = yHat.minus(y);

	    // Élévation au carré
	    SimpleMatrix squared = diff.elementMult(diff);

	    // Moyenne de toutes les valeurs
	    double sum = squared.elementSum();
	    return sum / (y.getNumRows() * y.getNumCols());
	}
	
	public void backpropagate(SimpleMatrix X, SimpleMatrix Y, double learningRate) {
	    Pair result = feed_forward(X);
	    SimpleMatrix yHat = (SimpleMatrix) result.getA();
	    @SuppressWarnings("unchecked")
		ArrayList<SimpleMatrix> activations = (ArrayList<SimpleMatrix>) result.getB();

	    // Calcul du gradient du coût
	    SimpleMatrix dA = yHat.minus(Y); // pour MSE et sigmoid en sortie : dA = (yHat - y)

	    // On démarre depuis la dernière couche
	    for (int i = layers.size() - 1; i >= 0; i--) {
	        SimpleMatrix A_prev = (i == 0) ? X : activations.get(i - 1);
	        dA = layers.get(i).backward(dA, A_prev, learningRate);
	    }
	}
	
	public void train(SimpleMatrix A0, SimpleMatrix Y, int epochs, double learningRate) {
	    for (int epoch = 0; epoch < epochs; epoch++) {
	    	
	        // Forward pass
	        Pair result = this.feed_forward(A0);
	        SimpleMatrix yHat = (SimpleMatrix) result.getA();

	        // Calcul du coût
	        double loss = this.cost(yHat, Y);

	        // Backpropagation (gère aussi la mise à jour des poids)
	        this.backpropagate(A0, Y, learningRate);
	        
	        if (epoch % 100 == 0) {
	            System.out.println("Époque " + epoch + " | Coût: " + loss);
	        }
	    }
	}
	
	public SimpleMatrix predict(SimpleMatrix input) {
	    Pair result = this.feed_forward(input);
	    return (SimpleMatrix) result.getA(); // La sortie finale (yHat)
	}
	
	public List<Float> predictList(SimpleMatrix input) {
		Pair result = this.feed_forward(input);
		SimpleMatrix output = (SimpleMatrix) result.getA();
		
		List<Float> predictions = new ArrayList<Float>();
		
		for (int i = 0; i < output.getNumElements(); i++) {
			predictions.add((float) output.get(i));
		}
		
		return predictions;
		
	}
	
	public int predictClass(SimpleMatrix input) {
	    SimpleMatrix output = this.predict(input);
	    
	    int maxIndex = 0;
	    double maxValue = output.get(0);

	    for (int i = 1; i < output.getNumElements(); i++) {
	        if (output.get(i) > maxValue) {
	            maxValue = output.get(i);
	            maxIndex = i;
	        }
	    }

	    return maxIndex;
	}
	
	public void save(String filename) throws IOException {
	    try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
	        out.writeInt(layers.size());
	        for (Layer layer : layers) {
	            out.writeObject(layer.getActiv()); // enum
	            out.writeObject(layer.getWeights().getDDRM());
	            out.writeObject(layer.getBias().getDDRM());
	        }
	    }
	}
	
	public void load(String filename) throws IOException, ClassNotFoundException {
	    layers.clear();

	    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
	        int numLayers = in.readInt();
	        for (int i = 0; i < numLayers; i++) {
	            ActivationFunction activ = (ActivationFunction) in.readObject();
	            DMatrixRMaj weights = (DMatrixRMaj) in.readObject();
	            DMatrixRMaj bias = (DMatrixRMaj) in.readObject();

	            Layer layer = new Layer(0, 0, activ); // tailles dummy
	            layer.setWeights(new SimpleMatrix(weights));
	            layer.setBias(new SimpleMatrix(bias));
	            layer.setActiv(activ);

	            layers.add(layer);
	        }
	    }
	}
	
	/**
	 * Permet de créer des enfants entre ce modèle et un autre et de les muter
	 * 
	 * @param other l'autre modèle parent
	 * @param proportion la proportion de mutation (entre 0 et 1), plus elle est grande, plus il y aura de mutation
	 * @param rate le taux de mutation (une valeur faible est préconisée)
	 * @param amount le nombre d'enfants
	 * @return une liste d'enfants mutés issues des 2 parents
	 */
	public List<NN> breedAndMutate(NN other, double proportion, double rate, int amount) {
		Random rand = new Random();

	    Layer l;
	    double mutation;
	    
	    List<NN> children = new ArrayList<>();
	    

	    for(int a = 0; a < amount; a++) {
	    	
	    	// At the beginning the child will be a clone of the parent A
		    NN child = clone();
		    
		    // For each layer of the child
		    for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
		    	l = layers.get(layerIndex);
		    	
		    	// For each weight of the layer
		    	for (int i = 0; i < l.getWeights().getNumRows(); i++) {
		    		for (int j = 0; j < l.getWeights().getNumCols(); j++) {
	
		    			/* 
		    			 * Which parent
		    			 * If nextBoolean is true -> This weight will be from parent B
		    			 * That means that the child have a 50% chance to inherit a weight from the parent B
		    			*/
		    			if(rand.nextBoolean()) { 
		    				
		    				// If we copy the weight from B, we should also copy its bias
		    				
		    				l.setWeight(i, j, other.getLayers().get(layerIndex).getWeight(i, j));
		    				l.setBias(i, other.getLayers().get(layerIndex).getBias(i));
		    			}
	
		    			// Should we mutate
		    			if(proportion >= rand.nextDouble()) {
		    				
		    				// Mutation using Gaussian distribution
		    				// If we mutate the weight, we should also mutate its bias
		    				
		    				mutation = rate * rand.nextGaussian();
		    				l.setWeight(i, j, l.getWeight(i, j) + mutation);
		    				mutation = rate * rand.nextGaussian();
		    				l.setBias(i, l.getBias(i) + mutation);
		    			}
		    		}
		    	}
		    }
		    
		    children.add(child);
	    }
	    
	    return children;
		
	}
	
	public NN clone() {
	    NN copy = new NN();
	    for (Layer layer : this.layers) {
	        copy.addLayer(layer.clone());
	    }
	    return copy;
	}
	
	@Deprecated
	public void mutate(double rate) {
	    for (Layer layer : layers) {
	        layer.mutate(rate);
	    }
	}

}
