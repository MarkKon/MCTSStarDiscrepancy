#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "mcts.hpp"

struct SingleHPStatistic {
	std::string algorithm_name;
	UCBHyperparameters hyperparameters;
	std::vector<double> values;

	SingleHPStatistic(std::string algorithm_name, UCBHyperparameters hyperparameters) : algorithm_name(algorithm_name), hyperparameters(hyperparameters) {}
	void addValue(double value) {
		values.push_back(value);
	}
};

struct HPStatistic {
	std::vector<SingleHPStatistic> statistics;

	HPStatistic() {}
	void addStatistic(std::string algorithm_name, UCBHyperparameters hyperparameters) {
		statistics.push_back(SingleHPStatistic(algorithm_name, hyperparameters));
	}
	void addSingle(SingleHPStatistic stat) {
		statistics.push_back(stat);
	}
	void addValue(std::string algorithm_name, UCBHyperparameters hyperparameters, double value) {
		for (auto& statistic : statistics) {
			if (statistic.algorithm_name == algorithm_name && statistic.hyperparameters == hyperparameters) {
				statistic.addValue(value);
				return;
			}
		}
		addStatistic(algorithm_name, hyperparameters);
		statistics.back().addValue(value);
	}
	double get_average(std::string algorithm_name, UCBHyperparameters hyperparameter) {
		double sum = 0.0;
        int count = 0;
        for (auto& statistic : statistics) {
            if (statistic.algorithm_name == algorithm_name && statistic.hyperparameters == hyperparameter) {
                for (auto& value : statistic.values) {
                    sum += value;
                    count++;
                }
            }
        }
        return sum / count;
    }
	// Function to output the statistics to a file
	void output_to_file(std::string filename) {
		std::ofstream file;
		file.open(filename);
		if (!file.is_open()) {
			std::cerr << "Error opening file " << filename << std::endl;
			return;
		}
		// Write the header
		file << "Algorithm;Hyperparameter (C);Values" << std::endl;
		for (auto& statistic : statistics) {
			file << statistic.algorithm_name << "; " << statistic.hyperparameters.c << "; ";
			for (auto& value : statistic.values) {
				file << value << ", ";
			}
			file << std::endl;
		}
		file.close();
	}
};

struct SingleTimeLineStatistic{
	std::vector<double> values;
	std::vector<int> iterations;

	SingleTimeLineStatistic() {}
	SingleTimeLineStatistic(std::vector<int> iterations) : iterations(iterations) {}
	void addValue(int iteration, double value) {
		iterations.push_back(iteration);
		values.push_back(value);
	}
};

struct TimeLineStatistic {
	std::vector<std::vector<double> > values;
	std::vector<int> iterations;
	unsigned int multiplicity;

	TimeLineStatistic() {}
	TimeLineStatistic(std::vector<int> iterations, unsigned int multiplicity) : iterations(iterations), multiplicity(multiplicity) {
		for (int i = 0; i < iterations.size(); i++) {
			values.push_back(std::vector<double>());
		}
	}
	void addValue(int iteration, std::vector<double> value) {
		iterations.push_back(iteration);
		values.push_back(value);
	}

	// Append a SingleTimeLineStatistic to the TimeLineStatistic
	void addSingle(SingleTimeLineStatistic stat) {
		// Check if the iterations are the same
		if (iterations.size() != stat.iterations.size()) {
			std::cerr << "Error: The number of iterations do not match" << std::endl;
			return;
		}
		// Add the values
		for (int i = 0; i < values.size(); i++) {
			values[i].push_back(stat.values[i]);
		}
	}

	// Function to output the statistics to a file
	void output_to_file(std::string filename) {
		std::ofstream file;
		file.open(filename);
		if (!file.is_open()) {
			std::cerr << "Error opening file " << filename << std::endl;
			return;
		}
		// Write the header
		file << "Iteration; Values " << std::endl;
		for (int i = 0; i < values.size(); i++) {
			file << iterations[i] << "; ";
			for (auto& value : values[i]) {
				file << value << ", ";
			}
			file << std::endl;
		}
		file.close();
	}
};



typedef std::tuple<std::vector<double>, double > PointAndValue;



void pointsAndValueToFile(std::string filename, std::vector<PointAndValue> pointsAndValues) {
	std::ofstream file;
	file.open(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file " << filename << std::endl;
		return;
	}
	// value; x1, x2 ...
	for (auto& pointAndValue : pointsAndValues) {
		file << std::get<1>(pointAndValue) << "; ";
		for (auto& value : std::get<0>(pointAndValue)) {
			file << value << ", ";
		}
		file << std::endl;
	}
	file.close();
}
