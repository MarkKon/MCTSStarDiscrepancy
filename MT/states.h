#pragma once
// All the states that are used in the MCTS algorithm

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


class Action {
public:
    unsigned int dimension;
    unsigned int value;

    Action() {
        dimension = 0;
        value = 0;
    };

    Action(unsigned int dim, unsigned int val) {
        dimension = dim;
        // value is either 0 or 1
        if (val > 1) {
			throw std::invalid_argument("Action value must be 0 or 1");
		}
		value = val;
    }

    Action(const Action& a) {
        dimension = a.dimension;
		value = a.value;
    }

    Action& operator=(const Action& a) {
        dimension = a.dimension;
        value = a.value;
        return *this;
    }

    bool operator==(const Action& a) {
        return dimension == a.dimension && value == a.value;
    }

    bool operator<(const Action& a) {
        if (dimension < a.dimension) {
			return true;
		}
        if (dimension > a.dimension) {
			return false;
		}
		return value < a.value;
    }
};

unsigned int bitlength(unsigned int n) {
    unsigned int h = 0;
    while (n > 0) {
        n >>= 1;
        h++;
    }
    return h;
}

unsigned int layercount(unsigned int n) {
    return n - pow(2, bitlength(n) - 1) + 1;
}


class GridState {
public:
    Grid* grid;

    std::vector<unsigned int> start;
    std::vector<unsigned int> end;

    unsigned int depth;

    GridState() {
        grid = nullptr;
        start = std::vector<unsigned int>();
        end = std::vector<unsigned int>();
        depth = 0;
    };
    GridState(Grid* g) {
        grid = g;
        // end is g-> gridDims() - 1 and start is all 0s
        end = g->gridDims();
        // subtract 1 from each element of end
        for (unsigned int i = 0; i < end.size(); i++) {
            end[i]--;
        }
        start = std::vector<unsigned int>(end.size(), 0);
        depth = 0;
    };
    GridState(Grid* g, const std::vector<unsigned int>& s, const std::vector<unsigned int>& e, unsigned int depth) {
        grid = g;
        start = s;
        end = e;
        this->depth = depth;
    };
    GridState& operator=(const GridState& s) {
        grid = s.grid;
        start = s.start;
        end = s.end;
        depth = s.depth;
        return *this;
    };
    std::vector<Action> actions() {
        std::vector<Action> actions;
        for (unsigned int i = 0; i < start.size(); i++) {
            if (end[i] - start[i] > 1) {
                actions.push_back(Action(i, 0));
                actions.push_back(Action(i, 1));
            }
        }
        return actions;
    };
    bool operator==(const GridState& s) const {
        // Grid reference is not checked
        return depth == depth && start == s.start && end == s.end;
    };
    bool operator<(const GridState& s) const {
        if (depth < s.depth) {
            return true;
        }
        if (depth > s.depth) {
            return false;
        }
        if (start < s.start) {
            return true;
        }
        if (start > s.start) {
            return false;
        }
        return end < s.end;
    };
    GridState operator+(const Action& a) {
        std::vector<unsigned int> new_start = start;
        std::vector<unsigned int> new_end = end;
        if (end[a.dimension] - start[a.dimension] < 2) {
            throw std::runtime_error("Cannot add action to state");
        }
        unsigned int mid = start[a.dimension] + ((unsigned int)(end[a.dimension] - start[a.dimension]) / 2);
        if (a.value == 0) {
            new_end[a.dimension] = mid;
        }
        else {
            new_start[a.dimension] = mid;
        }
        return GridState(grid, new_start, new_end, depth + 1);
    };
    std::tuple<std::vector<double>, std::vector<double>> bounds() {
        std::vector<double> lower;
        std::vector<double> upper;
        for (unsigned int i = 0; i < start.size(); i++) {
            lower.push_back(grid->grid[i][start[i]]);
            upper.push_back(grid->grid[i][end[i]]);
        }
        return std::make_tuple(lower, upper);
    };
    std::vector<double> sample(std::mt19937 & mt) {
        std::vector<double> point;
        for (unsigned int i = 0; i < start.size(); i++) {
            point.push_back(grid->grid[i][start[i]] + (grid->grid[i][end[i]] - grid->grid[i][start[i]]) * mt() / mt.max());
        }
        return point;
    };
    bool isLeaf() {
        for (unsigned int i = 0; i < start.size(); i++) {
            if (end[i] - start[i] > 1) {
                return false;
            }
        }
        return true;
    };
};

class GridStateExact : public GridState {
    // Inherit everything but overrride the actions function to only split in one dimension
    public:
        GridStateExact() : GridState() {};
        GridStateExact(Grid* g) : GridState(g) {};
        std::vector<Action> actions() {
            std::vector<Action> actions;
            // Only split in dimension this.depth% d
            unsigned int d = start.size();
            unsigned int dim = depth % d;
            if (end[dim] - start[dim] > 1) {
                actions.push_back(Action(dim, 0));
                actions.push_back(Action(dim, 1));
            }
            return actions;
        };
        // Automatic conversion to GridStateExact from GridState
        GridStateExact(const GridState& s) : GridState(s) {};

};

class GridStateImprovedSample : public GridState {
    // Inherit everything from GridState but override the sample function
    public:
        GridStateImprovedSample() : GridState() {};
        GridStateImprovedSample(Grid* g) : GridState(g) {};
        std::vector<double> sample(std::mt19937& mt) {
			std::vector<double> point;
            unsigned int d = start.size();
            double d_double = d;
            for (unsigned int i = 0; i < d; i++) {
				// Get a uniform random number between 0 and 1
                double u = (double) mt() / (double) mt.max();
                // Transform via the transformation function s -> ((ub^d - lb^d) * s + lb^d)^1/d 
                double lb = grid->grid[i][start[i]];
                double ub = grid->grid[i][end[i]];
                double new_point = pow((pow(ub, d_double) - pow(lb, d_double)) * u + pow(lb, d_double), 1.0 / d_double);
                point.push_back(new_point);
			}
			return point;
		};
		// Automatic conversion to SampleGridState from GridState
        GridStateImprovedSample(const GridState& s) : GridState(s) {};

};

class GridStateImprovedSplit : public GridState {
    // Inherit everything but overrride the + operator to disbalance the split
    public:
		GridStateImprovedSplit() : GridState() {};
		GridStateImprovedSplit(Grid* g) : GridState(g) {};
		GridStateImprovedSplit(Grid* g, const std::vector<unsigned int>& s, const std::vector<unsigned int>& e, unsigned int depth) : GridState(g, s, e, depth) {};
        GridStateImprovedSplit operator+(const Action& a) {
			std::vector<unsigned int> new_start = start;
			std::vector<unsigned int> new_end = end;
            if (end[a.dimension] - start[a.dimension] < 2) {
				throw std::runtime_error("Cannot add action to state");
			}
            int d = start.size();
            // split at the rounded down 1/2^1/d point
            // For the dimension that is split, the split point is grid[i][start[i]] + (grid[i][end[i]] - grid[i][start[i]]) * (1/2^1/d)
            // Then split at the last index that is smaller than this point
            double split_point = grid->grid[a.dimension][start[a.dimension]] + (grid->grid[a.dimension][end[a.dimension]] - grid->grid[a.dimension][start[a.dimension]]) / pow(2, 1.0 / d);
            unsigned int split_index = start[a.dimension];
            // Use binary search to find the split index
            unsigned int low = start[a.dimension];
            unsigned int high = end[a.dimension];
            while (low <= high) {
                unsigned int mid = low + (high - low) / 2;
                if (grid->grid[a.dimension][mid] < split_point) {
                    split_index = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            if (a.value == 0) {
				new_end[a.dimension] = split_index;
			}
            else {
				new_start[a.dimension] = split_index;
			}
            return GridStateImprovedSplit(grid, new_start, new_end, depth + 1);
		};
		// Automatic conversion to SampleGridState from GridState
		GridStateImprovedSplit(const GridState& s) : GridState(s) {};

};

class GridStateExactAndImprovedSplit : public GridStateImprovedSplit {
    // inherit everything from ImprovedSplit but take the actions function to only split in one dimension
    public:
        GridStateExactAndImprovedSplit() : GridStateImprovedSplit() {};
        GridStateExactAndImprovedSplit(Grid* g) : GridStateImprovedSplit(g) {};
        std::vector<Action> actions() {
            std::vector<Action> actions;
            // Only split in dimension this.depth% d
            unsigned int d = start.size();
            unsigned int dim = depth % d;
            if (end[dim] - start[dim] > 1) {
                actions.push_back(Action(dim, 0));
                actions.push_back(Action(dim, 1));
            }
            return actions;
        };
        // Automatic conversion to GridStateExact from GridState
        GridStateExactAndImprovedSplit(const GridState& s) : GridStateImprovedSplit(s) {};
};


class SampleGridState : public GridState {
    // inherit everything from GridState but override the sample function
public:
    SampleGridState() : GridState() {};
    SampleGridState(Grid* g) : GridState(g) {};
    std::vector<double> sample(std::mt19937 & mt) {
        std::vector<double> point;
        for (unsigned int i = 0; i < start.size(); i++) {
            point.push_back(grid->grid[i][start[i]] + (grid->grid[i][end[i]] - grid->grid[i][start[i]]) * mt() / mt.max());
        }
        return point;
    };
    // Automatic conversion to SampleGridState from GridState
    SampleGridState(const GridState& s) : GridState(s) {};
};

class SampleSmoothGridState : public GridState {
    // inherit everything from SampleGridState but:
    // Override the actions function to ensure that the splitting goes over all dimensions roughly equally
    // Add fields volume_weight and point_weight for BAST search
public:
    double volume_weight;
    double point_weight;
    double maxChildUCB;
    SampleSmoothGridState() : GridState() {
        volume_weight = 0;
		point_weight = 0;
        maxChildUCB = 0;
    }
    SampleSmoothGridState(Grid* g) : GridState(g) {
        volume_weight = calculate_volume_weight();
        point_weight = calculate_point_weight();
        maxChildUCB = 0;
    };
    double calculate_volume_weight() {
		// Get the bounds of the grid
		std::vector<double> lower;
		std::vector<double> upper;
		std::tie(lower, upper) = bounds();
		// Calculate the volume weight
		double upper_volume = 1;
		double lower_volume = 1;
		for (unsigned int i = 0; i < lower.size(); i++) {
			upper_volume *= upper[i];
			lower_volume *= lower[i];
		}
		return upper_volume - lower_volume;
	};
    double calculate_point_weight() {
		double point_weight = 0;
		for (unsigned int i = 0; i < start.size(); i++) {
			double diff = end[i] - start[i];
			if (diff > point_weight) {
				point_weight = diff;
			}
		}
		return point_weight;
	};

	std::vector<Action> actions() {
        // Get the dimension with the largest difference between start and end
        unsigned int max_diff_dim = 0;
        double max_diff = 0;
        for (unsigned int i = 0; i < start.size(); i++) {
            double diff = end[i] - start[i];
            if (diff > max_diff) {
                max_diff = diff;
				max_diff_dim = i;
            }
        }
        // Split only in the dimensions that have this difference value (or close to it)
        std::vector<Action> actions;
        for (unsigned int i = 0; i < start.size(); i++) {
            if ((end[i] - start[i] > max_diff - 1 ) && (end[i] - start[i] > 1)) {
				actions.push_back(Action(i, 0));
				actions.push_back(Action(i, 1));
			}
		}
        return actions;
	};
    // Automatic conversion to SampleGridState from GridState
    SampleSmoothGridState(const GridState& s) : GridState(s) {
        volume_weight = calculate_volume_weight();
		point_weight = calculate_point_weight();
        maxChildUCB = 0;
    };
    double getWeight() {
		return volume_weight + point_weight;
	};
};

class GridStateSmoothImprovedSplit : public GridStateImprovedSplit {
    // inherit everything from GridState but:
    // Override the actions function to ensure that the splitting goes over all dimensions roughly equally
    // Add field amount_split as a measure of how much the state has been split in each dimension
public:
    double volume_weight;
    double point_weight;
    double maxChildUCB;
    std::vector<unsigned int> amount_split;
    GridStateSmoothImprovedSplit() : GridStateImprovedSplit() {
        amount_split = std::vector<unsigned int>(start.size(), 0);
        volume_weight = 0;
        point_weight = 0;
        maxChildUCB = 0;
    };
    GridStateSmoothImprovedSplit(Grid* g) : GridStateImprovedSplit(g) {
        amount_split = std::vector<unsigned int>(start.size(), 0);
        volume_weight = calculate_volume_weight();
        point_weight = calculate_point_weight();
        maxChildUCB = 0;
    };
    GridStateSmoothImprovedSplit(Grid* g, const std::vector<unsigned int>& s, const std::vector<unsigned int>& e, unsigned int depth) : GridStateImprovedSplit(g, s, e, depth) {
        amount_split = std::vector<unsigned int>(start.size(), 0);
    };
    std::vector<Action> actions() {
        // Get the dimension with the smallest amount of splits
        unsigned int min_split_dim = 0;
        unsigned int min_split = std::numeric_limits<unsigned int>::max();
        for (unsigned int i = 0; i < start.size(); i++) {
            if (amount_split[i] < min_split) {
                min_split = amount_split[i];
                min_split_dim = i;
            }
        }
        // Split only in the dimensions that have this split value (or close to it)
        std::vector<Action> actions;
        for (unsigned int i = 0; i < start.size(); i++) {
            if (amount_split[i] == min_split && end[i] - start[i] > 1) {
                actions.push_back(Action(i, 0));
                actions.push_back(Action(i, 1));
            }
        }
        return actions;
    };
    // Automatic conversion to SampleGridState from GridState
    GridStateSmoothImprovedSplit(const GridState& s) : GridStateImprovedSplit(s) {
        amount_split = std::vector<unsigned int>(start.size(), 0);
    };
        double calculate_volume_weight() {
		// Get the bounds of the grid
		std::vector<double> lower;
		std::vector<double> upper;
		std::tie(lower, upper) = bounds();
		// Calculate the volume weight
		double upper_volume = 1;
		double lower_volume = 1;
		for (unsigned int i = 0; i < lower.size(); i++) {
			upper_volume *= upper[i];
			lower_volume *= lower[i];
		}
		return upper_volume - lower_volume;
	};
    double calculate_point_weight() {
		double point_weight = 0;
		for (unsigned int i = 0; i < start.size(); i++) {
			double diff = end[i] - start[i];
			if (diff > point_weight) {
				point_weight = diff;
			}
		}
		return point_weight;
	};
    double getWeight() {
		return volume_weight + point_weight;
	};
};




class LeftDeterministicGridState : public GridState {
    // inherit everything from GridState but override the sample function
public:
    LeftDeterministicGridState() : GridState() {};
    LeftDeterministicGridState(Grid* g) : GridState(g) {};
    std::vector<double> sample(std::mt19937 & mt) {
        std::vector<double> point;
        for (unsigned int i = 0; i < start.size(); i++) {
            point.push_back(grid->grid[i][start[i]]);
        }
        return point;
    };
    // Automatic conversion to LeftDeterministicGridState from GridState
    LeftDeterministicGridState(const GridState& s) : GridState(s) {};
};

class RightDeterministicGridState : public GridState {
    // inherit everything from GridState but override the sample function with the rightmost point - machine epsilon
public:
    RightDeterministicGridState() : GridState() {};
    RightDeterministicGridState(Grid* g) : GridState(g) {};
    std::vector<double> sample(std::mt19937 & mt) {
        std::vector<double> point;
        constexpr double epsilon = std::numeric_limits<double>::epsilon();
        for (unsigned int i = 0; i < start.size(); i++) {
            point.push_back(grid->grid[i][end[i]] - epsilon);
        }
        return point;
    };
    // Automatic conversion to RightDeterministicGridState from GridState
    RightDeterministicGridState(const GridState& s) : GridState(s) {};
};

class FixPointGridState : public GridState {
public:
    std::vector<double> point;
    unsigned int unsampled_dim;
    // Override the initialization function
    FixPointGridState() : GridState()
    {
        point = std::vector<double>();
        unsampled_dim = std::numeric_limits<unsigned int>::max();
    };
    FixPointGridState(Grid* g, std::mt19937 mt) : GridState(g)
    {
        point = std::vector<double>();
        for (unsigned int i = 0; i < start.size(); i++)
        {
            point.push_back(grid->grid[i][start[i]] + (grid->grid[i][end[i]] - grid->grid[i][start[i]]) * mt() / mt.max());
        }
        unsampled_dim = std::numeric_limits<unsigned int>::max();
    };
    FixPointGridState(Grid* g, std::vector<double> point, unsigned int sample_dim, std::vector<unsigned int> start,
        std::vector<unsigned int> end, unsigned int depth) : GridState(g, start, end, depth) {
        this->point = point;
        unsampled_dim = sample_dim;
    };
    // Override the sample function
    std::vector<double> sample(std::mt19937 & mt)
    {
        if (unsampled_dim == std::numeric_limits<unsigned int>::max())
        {
            return point;
        }
        else
        {
            point[unsampled_dim] = grid->grid[unsampled_dim][start[unsampled_dim]] + (grid->grid[unsampled_dim][end[unsampled_dim]] - grid->grid[unsampled_dim][start[unsampled_dim]]) * mt() / mt.max();
            return point;
        }
    };
    // Override the addition function
    FixPointGridState operator+(const Action& a)
    {
        std::vector<unsigned int> new_start = start;
        std::vector<unsigned int> new_end = end;
        if (end[a.dimension] - start[a.dimension] < 2) {
            throw std::runtime_error("Cannot add action to state");
        }
        unsigned int mid = start[a.dimension] + ((unsigned int)(end[a.dimension] - start[a.dimension]) / 2);
        if (a.value == 0) {
            new_end[a.dimension] = mid;
        }
        else {
            new_start[a.dimension] = mid;
        }
        return FixPointGridState(grid, point, a.dimension, new_start, new_end, depth + 1);
    };
};


//class FixDimSampleGridState : public GridState {
//	// inherit everything from GridState but override the sample function
//    public:
//        std::vector<double> 
        
