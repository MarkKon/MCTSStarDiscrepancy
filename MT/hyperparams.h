#pragma once
#include <vector>
#include <stdexcept>
#include <tuple>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <iostream>
#include <random>
#include "point_set.h"
#include "dag.h"
#include "states.h"
#include "mcts.h"
#include "statistics.h"

std::vector<UCBHyperparameters> cEquidistant(double c_min, double c_max, unsigned int n) {
	std::vector<UCBHyperparameters> params;
	for (int i = 0; i < n; i++) {
		double c = c_min + (c_max - c_min) * i / (n - 1);
		params.push_back(UCBHyperparameters(c));
	}
	return params;
}

std::vector<UCBHyperparameters> cLogEquidistant(double c_min, double c_max, unsigned int n) {
	std::vector<UCBHyperparameters> params;
	for (int i = 0; i < n; i++) {
		double c = c_min * std::pow(c_max / c_min, i / (n - 1.0));
		params.push_back(UCBHyperparameters(c));
	}
	return params;
}



template<class State, class UCBClass>
std::pair<UCBHyperparameters, double> treeHyperparameterSearch(std::vector<UCBHyperparameters> params, unsigned int its, unsigned int multiplicity, unsigned int point_n, unsigned int point_d) {
	double max_v = 0;
	UCBHyperparameters max_params;
	FaurePointSet pointSet(point_d, point_n);
	pointSet.generate();
#pragma omp parallel shared(max_v, max_params, pointSet) num_threads(10)
	{
#pragma omp for
		// Iterate over the muliplicity
		for (int i = 0; i < multiplicity; i++) {
			Grid grid(pointSet);
			State gridState(&grid);
			std::mt19937 mt(i);
			for (int j = 0; j < params.size(); j++) {
				UCBClass mctsgrid(&pointSet, gridState, its * point_d * 3, mt, params[j]);
				mctsgrid.run(its);
#pragma omp critical
				if (mctsgrid.maxValue() > max_v) {
					max_v = mctsgrid.maxValue();
					max_params = params[j];
				}
			}
		}
	}
	return std::pair<UCBHyperparameters, double>(max_params, max_v);
}

template<class State, class UCBClass>
std::pair<UCBHyperparameters, double> treeHyperparameterSearchFixedPoint(std::vector<UCBHyperparameters> params, unsigned int its, unsigned int multiplicity, unsigned int point_n, unsigned int point_d) {
	double max_v = 0;
	UCBHyperparameters max_params;
	FaurePointSet pointSet(point_d, point_n);
	pointSet.generate();
#pragma omp parallel shared(max_v, max_params, pointSet) num_threads(4)
	{
#pragma omp for
		// Iterate over the muliplicity
		for (int i = 0; i < multiplicity; i++) {
			Grid grid(pointSet);
			std::mt19937 mt(i);
			State gridState(&grid, mt);
			for (int j = 0; j < params.size(); j++) {
				UCBClass mctsgrid(&pointSet, gridState, its * point_d * 3, mt, params[j]);
				mctsgrid.run(its);
#pragma omp critical
				if (mctsgrid.maxValue() > max_v) {
					max_v = mctsgrid.maxValue();
					max_params = params[j];
				}
			}
		}
	}
	return std::pair<UCBHyperparameters, double>(max_params, max_v);
}


std::vector<std::tuple<std::string, UCBHyperparameters, double>> detRAllSearch(std::vector<UCBHyperparameters> params, unsigned int its, unsigned int multiplicity, unsigned int n, unsigned int d)
{
	std::vector<std::tuple<std::string, UCBHyperparameters, double>> results;
	auto out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSUCB1MaxGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetMax", out.first, out.second));
	out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSUCB1AMaxGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetAMax", out.first, out.second));
	out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSUCB1AvgGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetAvg", out.first, out.second));
	out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSUCB1SumGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetSum", out.first, out.second));
	out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSUCBMinGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetMin", out.first, out.second));
	out = treeHyperparameterSearch<RightDeterministicGridState, TreeMCTSBayesGridRDet>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("RDetBayes", out.first, out.second));
	return results;
}

std::vector<std::tuple<std::string, UCBHyperparameters, double>> gridStateAllSearch(std::vector<UCBHyperparameters> params, unsigned int its, unsigned int multiplicity, unsigned int n, unsigned int d)
{
	std::vector<std::tuple<std::string, UCBHyperparameters, double>> results;
	auto out = treeHyperparameterSearch<SampleGridState, TreeMCTSUCB1MaxGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("Max", out.first, out.second));
	out = treeHyperparameterSearch<SampleGridState, TreeMCTSUCB1AMaxGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("AMax", out.first, out.second));
	out = treeHyperparameterSearch<SampleGridState, TreeMCTSUCB1AvgGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("Avg", out.first, out.second));
	out = treeHyperparameterSearch<SampleGridState, TreeMCTSUCB1SumGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("Sum", out.first, out.second));
	out = treeHyperparameterSearch<SampleGridState, TreeMCTSUCBMinGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("Min", out.first, out.second));
	out = treeHyperparameterSearch<SampleGridState, TreeMCTSBayesGrid>(params, its, multiplicity, n, d);
	results.push_back(std::tuple<std::string, UCBHyperparameters, double>("Bayes", out.first, out.second));
	return results;
}



template<class State, class UCBClass>
SingleHPStatistic treeSingleSearchGrid(UCBHyperparameters params, unsigned int its, unsigned int multiplicity,
	unsigned int point_n, unsigned int point_d,	std::string name) {
	SingleHPStatistic statistic(name, params);
	FaurePointSet pointSet(point_d, point_n);
	pointSet.generate();
#pragma omp parallel shared(statistic, pointSet)
	{
#pragma omp for
		// Iterate over the muliplicity
		for (int i = 0; i < multiplicity; i++) {
			Grid grid(pointSet);
			State gridState(&grid);
			std::mt19937 mt(i);
			UCBClass mctsgrid(&pointSet, gridState, its * point_d * 3, mt, params);
			mctsgrid.run(its);
#pragma omp critical
			statistic.addValue(mctsgrid.maxValue());
		}
	}
	return statistic;
}

template<class State, class UCBClass>
SingleHPStatistic treeSingleSearchGridSobol(UCBHyperparameters params, unsigned int its, unsigned int multiplicity,
	unsigned int point_n, unsigned int point_d,	std::string name) {
	SingleHPStatistic statistic(name, params);
	SobolPointSet pointSet(point_d, point_n);
	pointSet.generate();
#pragma omp parallel shared(statistic, pointSet)
	{
#pragma omp for
		// Iterate over the muliplicity
		for (int i = 0; i < multiplicity; i++) {
			Grid grid(pointSet);
			State gridState(&grid);
			std::mt19937 mt(i);
			UCBClass mctsgrid(&pointSet, gridState, its * point_d * 3, mt, params);
			mctsgrid.run(its);
#pragma omp critical
			statistic.addValue(mctsgrid.maxValue());
		}
	}
	return statistic;
}


template<class State, class UCBClass>
SingleHPStatistic treeSingleSearchFixedPoint(UCBHyperparameters params, unsigned int its, unsigned int multiplicity,
	unsigned int point_n, unsigned int point_d,	std::string name) {
	SingleHPStatistic statistic(name, params);
	FaurePointSet pointSet(point_d, point_n);
	pointSet.generate();
#pragma omp parallel shared(statistic, pointSet)
	{
#pragma omp for
		// Iterate over the muliplicity
		for (int i = 0; i < multiplicity; i++) {
			Grid grid(pointSet);
			std::mt19937 mt(i);
			State gridState(&grid, mt);
			UCBClass mctsgrid(&pointSet, gridState, its * point_d * 3, mt, params);
			mctsgrid.run(its);
#pragma omp critical
			statistic.addValue(mctsgrid.maxValue());
		}
		
	}
	return statistic;
}

template<class State, class UCBClass>
HPStatistic treeOneSearchGrid(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int point_n, unsigned int point_d, std::string name) {
	HPStatistic statistic;
	for (auto& p : params) {
		statistic.addSingle(
			treeSingleSearchGrid<State, UCBClass>(p, its, multiplicity, point_n, point_d, name)
		);
	}
	return statistic;
}

template<class State, class UCBClass>
HPStatistic treeOneSearchFixedPoint(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int point_n, unsigned int point_d, std::string name) {
	HPStatistic statistic;
	for (auto& p : params) {
		statistic.addSingle(
			treeSingleSearchFixedPoint<State, UCBClass>(p, its, multiplicity, point_n, point_d, name)
		);
	}
	return statistic;
}


template<class State, class UCBClass>
UCBClass treeSearchwithRestarts(PointSet* p, State rootState, UCBHyperparameters hyperparameters, unsigned int its) {
	// For use its/ 5 to run 200 mcts with its/1000 iterations
	// initialise the mcts
	std::vector<UCBClass> trees;
	for (int i = 0; i < 200; i++) {
		UCBClass mcts = UCBClass(p, rootState, its, std::mt19937(i), hyperparameters);
		mcts.run(its / 1000);
		trees.push_back(mcts);
	}
	// Now select the 50 best trees
	std::sort(trees.begin(), trees.end(), [](UCBClass a, UCBClass b) { return a.maxValue() > b.maxValue(); });
	// Now run the best 50 trees for its/100 iterations
	for (int i = 0; i < 50; i++) {
		UCBClass mcts = trees[i];
		mcts.run(its / 100);
		trees[i] = mcts;
	}
	// Now select the best tree
	std::sort(trees.begin(), trees.end(), [](UCBClass a, UCBClass b) { return a.maxValue() > b.maxValue(); });
	// Run the best tree for 3its/10 iterations
	trees[0].run(3 * its / 10);
	return trees[0];
}






HPStatistic detRAllStatistic(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{

	HPStatistic statistic;
	for (auto& p : params) {
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCB1MaxGridRDet>(p, its, multiplicity, n, d, "RDetMax")
		);
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCB1AMaxGridRDet>(p, its, multiplicity, n, d, "RDetAMax")
		);
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCB1AvgGridRDet>(p, its, multiplicity, n, d, "RDetAvg")
		);
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCB1SumGridRDet>(p, its, multiplicity, n, d, "RDetSum")
		);
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCBMinGridRDet>(p, its, multiplicity, n, d, "RDetMin")
		);
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSBayesGridRDet>(p, its, multiplicity, n, d, "RDetBayes")
		);
	}
	return statistic;
};

HPStatistic gridStateAllStatistic(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{
	HPStatistic statistic;
	for (auto& p : params) {
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1MaxGrid>(p, its, multiplicity, n, d, "Max")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1AMaxGrid>(p, its, multiplicity, n, d, "AMax")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1AvgGrid>(p, its, multiplicity, n, d, "Avg")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSGreedyUCBMinGrid>(p, its, multiplicity, n, d, "GreedyAvg")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1SumGrid>(p, its, multiplicity, n, d, "Sum")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCBMinGrid>(p, its, multiplicity, n, d, "Min")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSGreedyUCBMinGrid>(p, its, multiplicity, n, d, "GreedyMin")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSBayesGrid>(p, its, multiplicity, n, d, "Bayes")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSGreedyBayesGrid>(p, its, multiplicity, n, d, "GreedyBayes")
		);
	}
	return statistic;
};

HPStatistic smoothGridStateAllStatistic(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{
	HPStatistic statistic;
	for (auto& p : params) {
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSUCB1MaxSmoothGrid>(p, its, multiplicity, n, d, "Max")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSUCB1AMaxSmoothGrid>(p, its, multiplicity, n, d, "AMax")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSUCB1AvgSmoothGrid>(p, its, multiplicity, n, d, "Avg")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSUCB1SumSmoothGrid>(p, its, multiplicity, n, d, "Sum")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSUCBMinSmoothGrid>(p, its, multiplicity, n, d, "Min")
		);
		statistic.addSingle(
			treeSingleSearchGrid<SampleSmoothGridState, TreeMCTSBayesSmoothGrid>(p, its, multiplicity, n, d, "Bayes")
		);
		}
	return statistic;
	};

HPStatistic allStatesUCTStatistic(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{
	HPStatistic statistic;
	for (auto& p : params) {
		// All states as in the thesis : Naive State Space, Grid State Space, Deterministic State Space, Improved Split State Space, Smooth State Space, Point State Space
		// Add Grid State Space
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "G")
		);
		std::cout << "Grid State Space done" << std::endl;
		// Add GridStateExact
		statistic.addSingle(
            treeSingleSearchGrid<GridStateExact, TreeMCTSUCB1Avg<GridStateExact, Action>>(p, its, multiplicity , n, d, "DisG")
		);
		std::cout << "Grid State Exact done" << std::endl;
        
		// Add Deterministic State Space
		statistic.addSingle(
			treeSingleSearchGrid<RightDeterministicGridState, TreeMCTSUCB1Avg<RightDeterministicGridState, Action>>(p, its, 1, n, d, "Det")
		);
		std::cout << "Deterministic State Space done" << std::endl;
		//Add Improved Split State Space
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSUCB1Avg<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "IS")
		);
		std::cout << "Improved Split State Space done" << std::endl;
		// Add Improved Exact State Space
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSUCB1Avg<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG")
		);
		std::cout << "Improved Exact Space done" << std::endl;
		// Add Point State Space
		statistic.addSingle(
			treeSingleSearchFixedPoint<FixPointGridState, TreeMCTSUCB1Avg<FixPointGridState, Action>>(p, its, multiplicity, n, d, "Pt")
		);
		std::cout << "Point State Space done" << std::endl;
	}
	return statistic;

};

HPStatistic allStatesUCTStatisticSobol(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{
	HPStatistic statistic;
	for (auto& p : params) {
		// // All states as in the thesis : Naive State Space, Grid State Space, Deterministic State Space, Improved Split State Space, Smooth State Space, Point State Space
		// // Add Grid State Space
		// statistic.addSingle(
		// 	treeSingleSearchGridSobol<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "G")
		// );
		// std::cout << "Grid State Space done" << std::endl;
		// Add GridStateExact
		statistic.addSingle(
            treeSingleSearchGridSobol<GridStateExact, TreeMCTSUCB1Avg<GridStateExact, Action>>(p, its, multiplicity , n, d, "DisG")
		);
		std::cout << "Grid State Exact done" << std::endl;
        
		// // Add Deterministic State Space
		// statistic.addSingle(
		// 	treeSingleSearchGridSobol<RightDeterministicGridState, TreeMCTSUCB1Avg<RightDeterministicGridState, Action>>(p, its, 1, n, d, "Det")
		// );
		// std::cout << "Deterministic State Space done" << std::endl;
		// //Add Improved Split State Space
		// statistic.addSingle(
		// 	treeSingleSearchGridSobol<GridStateImprovedSplit, TreeMCTSUCB1Avg<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "IS")
		// );
		// std::cout << "Improved Split State Space done" << std::endl;
		// Add Improved Exact State Space
		statistic.addSingle(
			treeSingleSearchGridSobol<GridStateExactAndImprovedSplit, TreeMCTSUCB1Avg<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG")
		);
		std::cout << "Improved Exact Space done" << std::endl;
		// // Add Point State Space
		// statistic.addSingle(
		// 	treeSingleSearchFixedPoint<FixPointGridState, TreeMCTSUCB1Avg<FixPointGridState, Action>>(p, its, multiplicity, n, d, "Pt")
		// );
		// std::cout << "Point State Space done" << std::endl;
	}
	return statistic;

};

HPStatistic GridvsImprovedUCTStatistic(std::vector<UCBHyperparameters> params, unsigned int its,
	unsigned int multiplicity, unsigned int n, unsigned int d)
{
	HPStatistic statistic;
	for (auto& p : params) {
		// Add Grid State Space
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "Grid State Space")
		);
		// Add Improved Split State Space
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSUCB1Avg<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "Improved Split State Space")
		);
	}
	return statistic;
};


TimeLineStatistic ImprovedSplitTimeline(UCBHyperparameters params, unsigned int its, unsigned int its_step, unsigned int multiplicity, unsigned int n, unsigned int d) {
	std::vector<int> iterations;
	for (int i = its_step; i <= its; i += its_step) {
		iterations.push_back(i);
	}
	TimeLineStatistic statistic(iterations, multiplicity);
	FaurePointSet pointSet(d, n);
	pointSet.generate();
	# pragma omp parallel shared(statistic, pointSet)
	{
	# pragma omp for
		for (int i = 0; i < multiplicity; i++) {
			SingleTimeLineStatistic single_stat;
			Grid grid(pointSet);
			GridStateImprovedSplit gridState(&grid);
			std::mt19937 mt(i);
			TreeMCTSUCB1Avg<GridStateImprovedSplit, Action> mctsgrid(&pointSet, gridState, its * d * 3, mt, params);
			for (int j = 0; j < iterations.size(); j++) {
				mctsgrid.run(its_step);
				single_stat.addValue(iterations[j], mctsgrid.maxValue());
			}
	# pragma omp critical
			statistic.addSingle(single_stat);
		}
	}
	return statistic;
};

TimeLineStatistic GridTimeline(UCBHyperparameters params, unsigned int its, unsigned int its_step, unsigned int multiplicity, unsigned int n, unsigned int d) {
	std::vector<int> iterations;
	for (int i = its_step; i <= its; i += its_step) {
		iterations.push_back(i);
	}
	TimeLineStatistic statistic(iterations, multiplicity);
	FaurePointSet pointSet(d, n);
	pointSet.generate();
	# pragma omp parallel shared(statistic, pointSet)
	{
	# pragma omp for
		for (int i = 0; i < multiplicity; i++) {
			SingleTimeLineStatistic single_stat;
			Grid grid(pointSet);
			GridState gridState(&grid);
			std::mt19937 mt(i);
			TreeMCTSUCB1Avg<GridState, Action> mctsgrid(&pointSet, gridState, its * d * 3, mt, params);
			for (int j = 0; j < iterations.size(); j++) {
				mctsgrid.run(its_step);
				single_stat.addValue(iterations[j], mctsgrid.maxValue());
			}
	# pragma omp critical
			statistic.addSingle(single_stat);
		}
	}
	return statistic;
};

HPStatistic PolicyCompare(std::vector<UCBHyperparameters>  params, unsigned int its, unsigned int multiplicity, unsigned int n, unsigned int d) {
	HPStatistic statistic;
	for (auto& p : params) {
		// All policies
		// Add Grid State Space
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/UCT")
		);
		std::cout << "G/UCT done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCBTuned<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/Tun")
		);
		std::cout << "G/Tun done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSSinglePlayer<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/SP")
		);
		std::cout << "G/SP done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCBDepth<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/Dep")
		);
		std::cout << "G/Dep done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSGreedyUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/Gre")
		);
		std::cout << "G/Gre done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSSqrtUCT<SampleGridState, Action>>(p, its, multiplicity, n, d, "G/Sqrt")
		);
		std::cout << "G/Sqrt done" << std::endl;
	}
	for (auto& p : params) {
		// All policies
		// Add Grid State Space
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSUCB1Avg<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/UCT")
		);
		std::cout << "ISDisG/UCT done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSUCBTuned<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/Tun")
		);
		std::cout << "ISDisG/Tun done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSSinglePlayer<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/SP")
		);
		std::cout << "ISDisG/SP done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSUCBDepth<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/Dep")
		);
		std::cout << "ISDisG/Dep done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSGreedyUCB1Avg<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/Gre")
		);
		std::cout << "ISDisG/Gre done" << std::endl;
		statistic.addSingle(
			treeSingleSearchGrid<GridStateExactAndImprovedSplit, TreeMCTSSqrtUCT<GridStateExactAndImprovedSplit, Action>>(p, its, multiplicity, n, d, "ISDisG/Sqrt")
		);
		std::cout << "ISDisG/Sqrt done" << std::endl;
	}

	return statistic;

}



HPStatistic GridPolicyCompare(std::vector<UCBHyperparameters>  params, unsigned int its, unsigned int multiplicity, unsigned int n, unsigned int d) {
	HPStatistic statistic;
	for (auto& p : params) {
		// The policies to compare: UCT, UCT-Tuned, SP-UCT, Depth, 1/2-Greedy+UCT, sqrtUCB + UCT, 
		// Add UCT
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "UCT")
		);
		std::cout << "UCT done" << std::endl;
		// Add UCT-Tuned
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCBTuned<SampleGridState, Action>>(p, its, multiplicity, n, d, "UCT-Tuned")
		);
		std::cout << "UCT-Tuned done" << std::endl;
		// Add SP-UCT
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSSinglePlayer<SampleGridState, Action>>(p, its, multiplicity, n, d, "SP-UCT")
		);
		std::cout << "SP-UCT done" << std::endl;
		// // Add Depth
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSUCBDepth<SampleGridState, Action>>(p, its, multiplicity, n, d, "Depth")
		);
		std::cout << "Depth done" << std::endl;
		// Add 1/2-Greedy+UCT
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSGreedyUCB1Avg<SampleGridState, Action>>(p, its, multiplicity, n, d, "1/2-Greedy+UCT")
		);
		std::cout << "1/2-Greedy+UCT done" << std::endl;
		// Add sqrtUCB + UCT
		statistic.addSingle(
			treeSingleSearchGrid<SampleGridState, TreeMCTSSqrtUCT<SampleGridState, Action>>(p, its, multiplicity, n, d, "sqrtUCB + UCT")
		);
		std::cout << "sqrtUCB + UCT done" << std::endl;
	}
	return statistic;
};

HPStatistic ImprovedSplitPolicyCompare(std::vector<UCBHyperparameters>  params, unsigned int its, unsigned int multiplicity, unsigned int n, unsigned int d) {
	HPStatistic statistic;
	for (auto& p : params) {
		// The policies to compare: UCT, UCT-Tuned, SP-UCT, Depth, 1/2-Greedy+UCT, sqrtUCB + UCT, 
		// Add UCT
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSUCB1Avg<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "UCT")
		);
		std::cout << "UCT done" << std::endl;
		// Add UCT-Tuned
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSUCBTuned<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "UCT-Tuned")
		);
		std::cout << "UCT-Tuned done" << std::endl;
		// // Add SP-UCT
		// statistic.addSingle(
		// 	treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSSinglePlayer<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "SP-UCT")
		// );
		// std::cout << "SP-UCT done" << std::endl;
		// // Add Depth
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSUCBDepth<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "Depth")
		);
		std::cout << "Depth done" << std::endl;
		// Add 1/2-Greedy+UCT
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSGreedyUCB1Avg<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "1/2-Greedy+UCT")
		);
		std::cout << "1/2-Greedy+UCT done" << std::endl;
		// Add sqrtUCB + UCT
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, TreeMCTSSqrtUCT<GridStateImprovedSplit, Action>>(p, its, multiplicity, n, d, "sqrtUCB + UCT")
		);
		std::cout << "sqrtUCB + UCT done" << std::endl;
		// Add BAST
		statistic.addSingle(
			treeSingleSearchGrid<GridStateImprovedSplit, BAST<GridStateSmoothImprovedSplit, Action>>(p, its, multiplicity, n, d, "BAST")
		);
		std::cout << "BAST done" << std::endl;
	}
	return statistic;
};