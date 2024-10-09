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

class kAction {
public:
    std::vector< unsigned int > dimensions;
    std::vector< unsigned int > values;

    kAction() {
        dimensions = std::vector<unsigned int>();
        values = std::vector<unsigned int>();
    }

    kAction(const std::vector<unsigned int>& d, const std::vector<unsigned int>& v) {
        dimensions = d;
        values = v;
    }

    kAction(const kAction& ka) {
        dimensions = ka.dimensions;
        values = ka.values;
    }

    kAction& operator=(const kAction& ka) {
        dimensions = ka.dimensions;
        values = ka.values;
        return *this;
    }

    bool operator==(const kAction& other) const {
        // Check if the dimension arrays have the same size
        if (dimensions.size() != other.dimensions.size()) {
            return false;
        }

        // Sort both dimension arrays for comparison
        std::vector<unsigned int> sorted_dimensions = dimensions;
        std::vector<unsigned int> sorted_other_dimensions = other.dimensions;
        std::sort(sorted_dimensions.begin(), sorted_dimensions.end());
        std::sort(sorted_other_dimensions.begin(), sorted_other_dimensions.end());

        // Compare the sorted dimension arrays
        if (sorted_dimensions != sorted_other_dimensions) {
            return false;
        }

        // Check if the values are equal when permuted according to the sorting of dimension
        std::vector<unsigned int> permuted_values(values.size());
        for (size_t i = 0; i < dimensions.size(); ++i) {
            permuted_values[i] = values[std::find(dimensions.begin(), dimensions.end(), other.dimensions[i]) - dimensions.begin()];
        }

        return permuted_values == other.values;
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

std::vector<std::vector<unsigned int>> generateBinaryVectors(unsigned int k) {
    std::vector<std::vector<unsigned int>> result;
    std::vector<unsigned int> current(k, 0);

    while (true) {
        result.push_back(current);

        // Find the rightmost 0
        int i = k - 1;
        while (i >= 0 && current[i] == 1) {
            i--;
        }

        // If all elements are 1, stop
        if (i < 0) {
            break;
        }

        // Flip the rightmost 0 and set all elements to its right to 0
        current[i] = 1;
        for (int j = i + 1; j < k; ++j) {
            current[j] = 0;
        }
    }

    return result;
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

template <int k>
class GridStateExactWithK {
public:
    Grid* grid;

    std::vector<unsigned int> start;
    std::vector<unsigned int> end;

    unsigned int depth;

    GridStateExactWithK() {
        grid = nullptr;
        start = std::vector<unsigned int>();
        end = std::vector<unsigned int>();
        depth = 0;
    };

    GridStateExactWithK(Grid* g) {
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

    GridStateExactWithK(Grid* g, const std::vector<unsigned int>& s, const std::vector<unsigned int>& e, unsigned int depth) {
        grid = g;
        start = s;
        end = e;
        this->depth = depth;
    };

    GridStateExactWithK& operator=(const GridStateExactWithK& s) {
        grid = s.grid;
        start = s.start;
        end = s.end;
        depth = s.depth;
        return *this;
    };


    std::vector<kAction> actions() {
        std::vector<kAction> actions;
        // Split in the dimensions this.depth * k % d to (this.depth + 1) * k - 1 % d
        unsigned int d = start.size();
        std::vector<unsigned int> dimarray;
        for(unsigned int i = 0; i < k; i++){
            unsigned int dim = (this->depth * k + i) % d;
            if (end[dim] - start[dim] > 1) {
                dimarray.push_back(dim);
            }
        }
        unsigned int shortK = dimarray.size();
        // Create all binary vectors
        std::vector<std::vector<unsigned int>> binaries = generateBinaryVectors(shortK);
        // For each binary, add the corresponding action
        for (auto& binary : binaries) {
            actions.push_back(kAction(dimarray, binary));
        }
        return actions;
    };

    bool operator==(const GridStateExactWithK& s) const {
        // Grid reference is not checked
        return depth == depth && start == s.start && end == s.end;
    };

    bool operator<(const GridStateExactWithK& s) const {
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

    GridStateExactWithK operator+(const kAction& a) {
        std::vector<unsigned int> new_start = start;
        std::vector<unsigned int> new_end = end;
        for(unsigned int i = 0; i < a.dimensions.size(); i++) {
            unsigned int dim = a.dimensions[i];
            if (end[dim] - start[dim] < 2) {
                throw std::runtime_error("Cannot add action to state");
            }
            unsigned int mid = start[dim] + ((unsigned int)(end[dim] - start[dim]) / 2);
            unsigned int value = a.values[i];
            if (value == 0) {
                new_end[dim] = mid;
            }
            else {
                new_start[dim] = mid;
            }
        }
        return GridStateExactWithK(grid, new_start, new_end, depth + 1);
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

template <int k>
class GridStateExactAndImprovedWithK : public GridStateExactWithK<k> {
// Inherit everything but overrride the + operator to disbalance the split
public:
    using GridStateExactWithK<k>::start;
    using GridStateExactWithK<k>::end;
    using GridStateExactWithK<k>::grid;
    using GridStateExactWithK<k>::depth;
    GridStateExactAndImprovedWithK() : GridStateExactWithK<k>() {};
    GridStateExactAndImprovedWithK(Grid* g) : GridStateExactWithK<k>(g) {};
    GridStateExactAndImprovedWithK(Grid* g, const std::vector<unsigned int>& s, const std::vector<unsigned int>& e, unsigned int depth) : GridStateExactWithK<k>(g, s, e, depth) {};
    GridStateExactAndImprovedWithK operator+(const kAction& a) {
        std::vector<unsigned int> new_start = start;
        std::vector<unsigned int> new_end = end;
        unsigned int d = start.size();
        for(unsigned int i = 0; i < a.dimensions.size(); i++) {
            unsigned int dim = a.dimensions[i];
            double split_point = grid->grid[dim][start[dim]] + (grid->grid[dim][end[dim]] - grid->grid[dim][start[dim]]) / pow(2, 1.0 / d);
            unsigned int split_index = start[dim];
            // Use binary search to find the split index
            unsigned int low = start[dim];
            unsigned int high = end[dim];
            while (low <= high) {
                unsigned int mid = low + (high - low) / 2;
                if (grid->grid[dim][mid] < split_point) {
                    split_index = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            if (a.values[dim] == 0) {
                new_end[dim] = split_index;
            }
            else {
                new_start[dim] = split_index;
            }
        }
        return GridStateExactWithK<k>(grid, new_start, new_end, depth + 1);
    };
    // Automatic conversion to SampleGridState from GridState
    GridStateExactAndImprovedWithK(const GridStateExactWithK<k>& s) : GridStateExactWithK<k>(s) {};

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
        
