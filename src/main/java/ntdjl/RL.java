package ntdjl;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import ntdjl.utils.ActivationFunction;
import ntdjl.utils.Genetics;

public class RL {
	
	HashMap<NN, Integer> agents;
	
	int[] structure;
	ActivationFunction fn;
	
	NN topOne, topTwo, topThree;
	
	int AB, BC, AC;
	
	double proportion, rate;
	
	int batch;
	
	public RL(int[] structure, ActivationFunction fn, double proportion, double rate, int AB, int BC, int AC) {
		this.structure = structure;
		this.fn = fn;
		
		List<NN> models = Genetics.createRandomModels(structure, fn, AB+BC+AC);
		
		this.agents = new HashMap<NN, Integer>();
		
		for(NN model : models) {
			this.agents.put(model, 0);
		}
		
		this.AB = AB;
		this.BC = BC;
		this.AC = AC;
		
		this.proportion = proportion;
		this.rate = rate;
		
		this.batch = 0;
	}
	
	public void nextGeneration() {
		NN top1 = null, top2 = null, top3 = null;
		int score1 = -1, score2 = -1, score3 = -1;
		
		for(NN model : agents.keySet()) {
			int score = agents.get(model);
			
			if(score > score1) {
				top1 = model;
				score1 = score;
			} else if(score > score2) {
				top2 = model;
				score2 = score;
			} else if(score > score3) {
				top3 = model;
				score3 = score;
			}
		}
		
		this.topOne = top1;
		this.topTwo = top2;
		this.topThree = top3;
		
		List<NN> ABmodels = top1.breedAndMutate(top2, proportion, rate, AB-1);
		List<NN> BCmodels = top2.breedAndMutate(top3, proportion, rate, BC-1);
		List<NN> ACmodels = top1.breedAndMutate(top3, proportion, rate, AC-1);
		
		agents.clear();
		
		// Keeping parents
		agents.put(top1, 0);
		agents.put(top2, 0);
		agents.put(top3, 0);
		
		for(NN model : ABmodels) this.agents.put(model, 0);
		for(NN model : BCmodels) this.agents.put(model, 0);
		for(NN model : ACmodels) this.agents.put(model, 0);
		
		
		this.batch++;
	}
	
	public Set<NN> getAgents() {
		return agents.keySet();
	}
	
	public int getScore(NN model) {
		if(this.agents.containsKey(model)) return this.agents.get(model);
		return -1;
	}
	
	public void setScore(NN model, int score) {
		agents.put(model, score);
	}
	
	
	public void saveModel(String filepath) {
		try {
			this.topOne.save(filepath + "_topOne.ntdjl");
			this.topTwo.save(filepath + "_topTwo.ntdjl");
			this.topThree.save(filepath + "_topThree.ntdjl");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void loadModel(String filepath) {
		try {
			this.topOne = new NN();
			this.topOne.load(filepath + "_topOne.ntdjl");
			
			this.topTwo = new NN();
			this.topTwo.load(filepath + "_topTwo.ntdjl");
			
			this.topThree = new NN();
			this.topThree.load(filepath + "_topThree.ntdjl");
			
			this.agents.clear();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}