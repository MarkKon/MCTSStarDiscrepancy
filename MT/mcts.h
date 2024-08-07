// mcts.h
#ifndef MCTS_HPP
#define MCTS_HPP

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

struct UCBHyperparameters {
	double c;
    UCBHyperparameters() : c(1.0) {};
	UCBHyperparameters(double c) : c(c) {};
    std::string toString() {
		return "UCBHyperparameters(c=" + std::to_string(c) + ")";
	}
    bool operator==(const UCBHyperparameters& other) const {
		return c == other.c;
	}
};


template <class State, class Action>
class DAGMCTS {
    // This class is a MCTS that is designed to be as general as possible.
public:
    PointSet* point_set;
    Grid grid;
    Grid* gridpointer;
    DAG<State, Action> tree;
    UCBHyperparameters hyperparameters;
    // random device is stored as a member variable to avoid reseeding
    std::mt19937 mt;


    DAGMCTS() {
		point_set = nullptr;
        grid = Grid();
        gridpointer = &grid;
        tree = DAG<State, Action>();
	}

    DAGMCTS(PointSet* p, State root, const unsigned int allocation_size, std::mt19937 twister, UCBHyperparameters hyperparameters) {
		point_set = p;
        grid = Grid(*point_set);
        gridpointer = &grid;
        tree = DAG<State, Action>(root, allocation_size);
        mt = twister;
        this->hyperparameters = hyperparameters;
    }

    virtual double rate(State parent, State child) {
        // This function should be implemented by the specific MCTS, throw error if not implemented
        throw std::runtime_error("rate function not implemented");
        return 0;
	}

    Action select(State state) {
        Package<State, Action> package = tree.getPackage(state);
        // Throw error if leaf node
        if (package.leaf) {
            throw std::runtime_error("Cannot select from leaf node");
		}
        // Throw error if no actions
        if (package.actions.size() == 0) {
            throw std::runtime_error("No actions to select from");
        }
        // In parallel, get the corresponding rate for each action 
        double max_value = -std::numeric_limits<double>::infinity();
        Action max_action = package.actions[0].first;
        #pragma omp parallel for reduction(max:max_value) shared(max_action)
        for (int i = 0; i < package.actions.size(); i++) {
			std::pair<Action,State> action_state = package.actions[i];
			double value = rate(state, action_state.second);
			if (value > max_value) {
				max_value = value;
				max_action = action_state.first;
			}
		}
        return max_action;
    }

    void expand(State state) {
        tree.addState(state);
	}

    double simulate(State state) {
        std::vector<double> point = state.sample(mt);
        return point_set->discrepancy_snapped(point, *gridpointer);
    }

    void backpropagate(double value, Path<State, Action> path ) {
        while (path.size() > 0) {
			State state = path.back().first;
			path.pop_back();
            tree.addVisit(state, value);
		}
	}

    Path<State, Action> selectPath(State state) {
		Path<State, Action> path;
        Action action;
        while (tree.states.find(state) != tree.states.end() && !tree.isLeaf(state)) {
			action = select(state);
			path.push_back(state, action);
			state = state + action;
		}
        path.push_back(state, action);
		return path;
	}

    void run(unsigned int iterations) {
        for (unsigned int i = 0; i < iterations; i++) {
			Path<State,Action> path = selectPath(tree.root);
            // Simulate the last state in the path
            State state = path.back().first;
            // If the state is already done, backpropagate the value and remove the action from the parent
            if (tree.isDone(state)) {
                double value = tree.getMax(state);
                backpropagate(value, path);
                // Remove action from the parent
                State parent = path.path[path.size() - 2].first;
                Action action = path.path[path.size() - 2].second;
                tree.removeAction(parent, action);
				continue;
			}
			expand(state);
			double value = simulate(state);
			backpropagate(value, path);
            // if the state is a leaf, set it as done
            if (state.isLeaf()) {
                tree.setDone(state);
                // Remove action from the parent
                State parent = path.path[path.size() - 2].first;
                Action action = path.path[path.size() - 2].second;
                tree.removeAction(parent, action);
            }
		}
    }

    double maxValue() {
		// The max value will be at the root due to the backpropagation
        return tree.getMax(tree.root);
	}

};

template<class State, class Action>
class TreeMCTS {
public:
    PointSet* point_set;
    Grid grid;
    Grid* gridpointer;
    std::vector<TreeNode<State, Action>> nodes;
    TreeNode<State, Action> root;
    UCBHyperparameters hyperparameters;
    std::mt19937 mt;
    std::vector<float> best_point;
    bool point_output;
    std::vector<std::tuple<std::vector<double>, double>> points_and_values;

    TreeMCTS() {
        point_set = nullptr;
		grid = Grid();
		gridpointer = &grid;
		nodes = std::vector<TreeNode<State, Action>>();
		root = nullptr;
        best_point = std::vector<float>();
        point_output = false;
        points_and_values = std::vector<std::tuple<std::vector<double>, double>>();
    }

    TreeMCTS(PointSet* p, State rootstate, const unsigned int allocation_size, std::mt19937 twister, UCBHyperparameters hyperparameters) {
        point_set = p;
        grid = Grid(*point_set);
        gridpointer = &grid;
        nodes = std::vector<TreeNode<State, Action>>();
        // Allocate space for the nodes
        nodes.reserve(allocation_size);
        TreeNode<State, Action> rootnode(rootstate);
        root = rootnode;
        mt = twister;
        this->hyperparameters = hyperparameters;
        point_output = false;
        points_and_values = std::vector<std::tuple<std::vector<double>, double>>();
    }

    virtual double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        // This function should be implemented by the specific MCTS, throw error if not implemented
        throw std::runtime_error("rate function not implemented");
        return 0;
    }

    Action select(TreeNode<State, Action> n)
    {
        // Throw error if leaf node
        if (n.leaf) {
            throw std::runtime_error("Cannot select from leaf node");
        }
        // Throw error if no actions
        if (n.actions.size() == 0) {
            throw std::runtime_error("No actions to select from");
        }
        Action max_action = n.actions[0].first;
        double max_value = -std::numeric_limits<double>::infinity();
        for (auto action_node : n.actions) {
            // Get Action and Node
            Action action = action_node.first;
            // Second is a pointer to the child node
            TreeNode<State, Action>* child = action_node.second;
            // If the child is unvisited, return it
            if (child->visits == 0) {
                return action;
            }
            // Calculate the rate
            double value = this->rate(n, *child);
            if (value > max_value) {
                max_value = value;
                max_action = action;
            }
        }
        return max_action;
    }

    void expand(TreeNode<State, Action>& n) {
        // Get the state of the node
        State state = n.state;
        // Calculate the children
        std::vector<Action> actions = state.actions();
        for (auto action : actions) {
            TreeNode<State, Action> child_node(state + action);
            nodes.push_back(child_node);
            // Add the child to the parent (if it is not already there)
            if (n.getChild(action) == nullptr) {
                n.addChild(action, &nodes.back());
            }
        }
    }

    double simulate(TreeNode<State, Action> n) {
        std::vector<double> point = n.state.sample(mt);
        double value = point_set->discrepancy_snapped(point, *gridpointer);
        if(point_output) {
            std::tuple<std::vector<double>, double> pvtuple = std::make_tuple(point, value);
            points_and_values.push_back(pvtuple);
        }
        return value;
    }

    void backpropagate(double value, NodePath<State, Action> path) {
        while (path.size() > 0) {
            TreeNode<State, Action>* node = path.back().first;
            path.pop_back();
            node->visits++;
            // Update standard deviation
            double mean_old = node->sum / (node->visits - 1);
            node->sum += value;
            double mean_new = node->sum / node->visits;
            // Update standard deviation as sd_new = sd_old + (value - mean_old) * (value - mean_new)
            node->stddev += (value - mean_old) * (value - mean_new);
            if (value > node->max) {
                node->max = value;
            }
        }
    }

    // Backpropagate the done value i.e. remove the action from the parent and if that parent becomes actionless, remove the action from its parent etc
    void backpropagateDone(NodePath<State, Action> path) {
        while (path.size() > 0) {
            TreeNode<State, Action>* node = path.back().first;
            path.pop_back();
            if (node->done || node->state.isLeaf()) {
                // Remove action from the parent
                TreeNode<State, Action>* parent = path.back().first;
                Action action = path.back().second;
                parent->removeChild(action);
                // If the parent is now actionless, set parent as done
                if (parent->actions.size() == 0) {
                    parent->done = true;
                }
                else
                {
                    break;
                }
            }
        }
    }

    NodePath<State, Action> selectPath(TreeNode<State, Action>* n) {
        NodePath<State, Action> path;
        Action action;
        TreeNode<State, Action>* node_ptr = n;
        while (!node_ptr->leaf && node_ptr->actions.size() > 0) {
            action = select(*node_ptr);
            path.push_back(node_ptr, action);
            node_ptr = node_ptr->getChild(action);
            }
        path.push_back(node_ptr, action);        
        return path;
    }

    void run(unsigned int iterations) {
        for (unsigned int i = 0; i < iterations; i++) {
			NodePath<State, Action> path = selectPath(&root);
			// Simulate the last state in the path
            TreeNode<State, Action>* node = path.back().first;
			// If the state is already done, backpropagate the value and remove the action from the parent
            if (node->done) {
				double value = node->max;
				backpropagate(value, path);
				// Remove action from the parent
                TreeNode<State, Action>* parent = path.path[path.size() - 2].first;
				Action action = path.path[path.size() - 2].second;
				parent->removeChild(action);
				continue;
			}
            expand(*node);
            double value = simulate(*node);
            backpropagate(value, path);
            // if the state is a leaf, set it as done
            if (node->state.isLeaf()) {
				node->done = true;
				// Remove action from the parent
                TreeNode<State, Action>* parent = path.path[path.size() - 2].first;
				Action action = path.path[path.size() - 2].second;
				parent->removeChild(action);
			}
            backpropagateDone(path);
		}
	}

    double maxValue() {
		// The max value will be at the root due to the backpropagation
		return root.max;
	}

};



template <class State, class Action>
class MCTSUCB1Max : public DAGMCTS<State, Action> {
    public:
	using DAGMCTS<State, Action>::DAGMCTS;

    // Override the rate function to implement UCB1
    double rate(State parent, State child) {
        // Get the total number of visits to the state
        unsigned int total_visits = this->tree.getVisits(parent);
        if (this->tree.states.find(child) == this->tree.states.end()) {
            return std::numeric_limits<double>::infinity();
        }
        // If child is done, return negative infinity
        if (this->tree.isDone(child)) {
			return -std::numeric_limits<double>::infinity();
		}
        // Get the number of visits to the new state
        unsigned int visits = this->tree.getVisits(child);
        // Get the value of the new state
        double value = this->tree.getMax(child);
        // Calculate the UCB1 value
        return value + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
    }
};

template <class State, class Action>
class MCTSUCB1Avg : public DAGMCTS<State, Action> {
	public:
	using DAGMCTS<State, Action>::DAGMCTS;

	// Override the rate function to implement UCB1
    double rate(State parent, State child) {
		// Get the total number of visits to the state
		unsigned int total_visits = this->tree.getVisits(parent);
        if (this->tree.states.find(child) == this->tree.states.end()) {
			return std::numeric_limits<double>::infinity();
		}
		// If child is done, return negative infinity
        if (this->tree.isDone(child)) {
			return -std::numeric_limits<double>::infinity();
		}
		// Get the number of visits to the new state
		unsigned int visits = this->tree.getVisits(child);
		// Get the value of the new state
		double value = this->tree.getSum(child);
		// Calculate the UCB1 value
		return value/visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};


template <class State, class Action>
class TreeMCTSUCB1Max : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;

	// Override the rate function to implement UCB1
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
        unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.max;
		return value + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};

template <class State, class Action>
class TreeMCTSUCB1AMax : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;

	// Override the rate function to implement UCB1
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.max;
		return value/visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};

template <class State, class Action>
class TreeMCTSUCB1Sum : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
		return value + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};

template <class State, class Action>
class TreeMCTSUCB1Avg : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
        unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
		return value/visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};

template <class State, class Action>
class TreeMCTSUCBMin : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
        return value / visits + this->hyperparameters.c / visits;
	}
};

template <class State, class Action>
class TreeMCTSBayes : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
            return -std::numeric_limits<double>::infinity();
        }
        unsigned int total_visits = parent.visits;
        unsigned int visits = child.visits;
        double value = child.sum;
        double sigma = std::sqrt(child.stddev / visits);
        return value / visits + this->hyperparameters.c * sigma / std::sqrt(visits);
    }
};

template <class State, class Action>
class TreeMCTSMyMax : public TreeMCTS<State, Action> {
	public:
	using TreeMCTS<State, Action>::TreeMCTS;
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
        return value/visits + this->hyperparameters.c * std::sqrt(2 * std::log(2 * visits));
	}
};

template <class State, class Action>
class TreeMCTSSinglePlayer : public TreeMCTS<State, Action> {
    public:
    using TreeMCTS<State, Action>::TreeMCTS;
    // Override the rate function to implement UCB1
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
            return -std::numeric_limits<double>::infinity();
        }
        unsigned int total_visits = parent.visits;
        unsigned int visits = child.visits;
        double value = child.sum;
        return value / visits + this->hyperparameters.c * (std::sqrt(std::log(total_visits) / visits) + std::sqrt((child.stddev + 1) / visits));
    }
};

template <class State, class Action>
class TreeMCTSUCBTuned : public TreeMCTS<State, Action> {
    public:
    using TreeMCTS<State, Action>::TreeMCTS;
    // Override the rate function to implement UCB1
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
            return -std::numeric_limits<double>::infinity();
        }
        unsigned int total_visits = parent.visits;
        unsigned int visits = child.visits;
        double value = child.sum;
        double v_j_n = child.stddev + std::sqrt(2* std::log(total_visits)/visits);
        return value / visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits * std::min(0.25, v_j_n));
    }
};

template <class State, class Action>
class TreeMCTSUCBDepth : public TreeMCTS<State, Action> {
    public:
    using TreeMCTS<State, Action>::TreeMCTS;
    // Override the rate function to implement UCB1
    double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
        if (child.done) {
            return -std::numeric_limits<double>::infinity();
        }
        unsigned int total_visits = parent.visits;
        unsigned int visits = child.visits;
        double value = child.sum;
        // Get the dimensions of the state
        unsigned int depth = child.state.depth;
        // Total expected depth is d* log2(n) where d is dimension
        unsigned int d = this->point_set->d;
        unsigned int n = this->point_set->n;
        double expected_depth = d * std::log2(n);
        double scaling_factor = (expected_depth/(2* expected_depth - depth));
        // Calculate the UCB1 value
        return value / visits + this->hyperparameters.c * std::pow(std::sqrt(std::log(total_visits) / visits), scaling_factor);

    }
};

template <class State, class Action>
class TreeMCTSGreedy : public TreeMCTS<State, Action> {
public:
    using TreeMCTS<State, Action>::TreeMCTS;
    // Override the select function: On the first layer (i.e. depth = 0) use greedy selection
    Action select(TreeNode<State, Action> n) {
        if (n.depth == 0) {
            // With probability 1/2 just select the maximum value node, with probability 1/2 select a random other node
            int K = n.actions.size();
            // Use mt to generate a random number between 0 and 1
            float d = std::uniform_real_distribution<float>(0, 1)(this->mt);
            // f d < 0.5, select the maximum value node
            if (d < 0.5) {
                Action maxAction = n.actions[0].first;
                double maxValue = -std::numeric_limits<double>::infinity();
                for (auto action_node : n.actions) {
                    // if unvisited, return it
                    if (action_node.second->visits == 0) {
                        return action_node.first;
                    }
                    double value = this->rate(n, *action_node.second);
                    if (value > maxValue) {
                        maxValue = value;
                        maxAction = action_node.first;
                    }
                }
                return maxAction;
            }
            // If d >= 0.5, select a random node
            else {
                std::uniform_int_distribution<int> dist(0, K - 1);
                return n.actions[dist(this->mt)].first;
            }
        }
        // If not on the first layer, use the parent selection
        else {
            return TreeMCTS<State, Action>::select(n);
        }
    }
};

template <class State, class Action>
class TreeMCTSGreedyUCB1Avg : public TreeMCTSGreedy<State, Action> {
	public:
	using TreeMCTSGreedy<State, Action>::TreeMCTSGreedy;
	// Override the rate function to implement UCB1
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
		return value / visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits);
	}
};

template <class State, class Action>
class TreeMCTSSqrtUCT : public TreeMCTSUCB1Avg<State, Action> {
    public:
    using TreeMCTSUCB1Avg<State, Action>::TreeMCTSUCB1Avg;
    // Override the select function: On the first layer (i.e. depth = 0) use UCB-Sqrt
    Action select(TreeNode<State, Action> n) {
        if (n.depth == 0) {
            // Use UCB-Sqrt, i.e. select the action with the maximum value of value/visits + c * sqrt(sqrt(total_visits)/visits)
            Action maxAction = n.actions[0].first;
            double maxValue = -std::numeric_limits<double>::infinity();
            for (auto action_node : n.actions) {
                // if unvisited, return it
                if (action_node.second->visits == 0) {
                    return action_node.first;
                }
                unsigned int total_visits = n.visits;
                unsigned int visits = action_node.second->visits;
                double value = action_node.second->sum;
                double rate = value / visits + this->hyperparameters.c * std::sqrt(std::sqrt(total_visits) / visits);
                if (rate > maxValue) {
                    maxValue = rate;
                    maxAction = action_node.first;
                }
            }
            return maxAction;
        }
        // If not on the first layer, use the parent selection
        else {
            return TreeMCTS<State, Action>::select(n);
        }
    }
};

template <class State, class Action>
class TreeMCTSGreedyUCBMin : public TreeMCTSGreedy<State, Action> {
	public:
	using TreeMCTSGreedy<State, Action>::TreeMCTSGreedy;
	// Override the rate function to implement UCB1
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
		return value / visits + this->hyperparameters.c / visits;
	}
};

template <class State, class Action>
class TreeMCTSGreedyBayes : public TreeMCTSGreedy<State, Action> {
    public:
	using TreeMCTSGreedy<State, Action>::TreeMCTSGreedy;
	// Override the rate function to implement UCB1
	double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
		if (child.done) {
			return -std::numeric_limits<double>::infinity();
		}
		unsigned int total_visits = parent.visits;
		unsigned int visits = child.visits;
		double value = child.sum;
		double sigma = std::sqrt(child.stddev / visits);
		return value / visits + this->hyperparameters.c * sigma / std::sqrt(visits);
	}
};

template <class State, class Action>
class BAST : public TreeMCTS<State, Action> {
    // Redefine the rate function to implement the BAST algorithm with the weights
    // This only works if the State class has a method double getWeight() and a member maxChildUCB that will be kept up to date in the backpropagation
    // 
    public:
        using TreeMCTS<State,Action>::root;
        using TreeMCTS<State,Action>::TreeMCTS;
        using TreeMCTS<State, Action>::expand;
        using TreeMCTS<State, Action>::simulate;
        using TreeMCTS<State, Action>::selectPath;
        void backpropagate(double value, NodePath<State, Action> path) {
            double maxUCB = 0;
            while (path.size() > 0) {
				TreeNode<State, Action>* node = path.back().first;
				path.pop_back();
				node->visits++;
                double mean_old = node->sum / (node->visits - 1);
				node->sum += value;
                double mean_new = node->sum / node->visits;
				// Update standard deviation as sd_new = sd_old + (value - mean_old) * (value - mean_new)
                node->stddev += (value - mean_old) * (value - mean_new);

                if (value > node->max) {
					node->max = value;
				}
                double maxChildUCB = node->state.maxChildUCB;
				// Update the maxChildUCB
                if (maxChildUCB > maxUCB) {
					maxUCB = maxChildUCB;
				}
                else {
                    node->state.maxChildUCB = maxUCB;
                }
			}
		}
        // Redefine run to set the maxChildUCB of the expanded node
        void run(unsigned int iterations) {
            for (unsigned int i = 0; i < iterations; i++) {
                NodePath<State, Action> path = selectPath(&root);
                // Simulate the last state in the path
                TreeNode<State, Action>* node = path.back().first;
                // If the state is already done, backpropagate the value and remove the action from the parent
                if (node->done) {
                    double value = node->max;
                    backpropagate(value, path);
                    // Remove action from the parent
                    TreeNode<State, Action>* parent = path.path[path.size() - 2].first;
                    Action action = path.path[path.size() - 2].second;
                    parent->removeChild(action);
                    continue;
                }
                expand(*node);
                double value = simulate(*node);
                double ucb = value + this->hyperparameters.c * std::sqrt(2 * std::log(node->visits));
                node->state.maxChildUCB = ucb;
                backpropagate(value, path);
                // if the state is a leaf, set it as done
                if (node->state.isLeaf()) {
                    node->done = true;
                    // Remove action from the parent
                    TreeNode<State, Action>* parent = path.path[path.size() - 2].first;
                    Action action = path.path[path.size() - 2].second;
                    parent->removeChild(action);
                }
            }
        }
        double rate(TreeNode<State, Action> parent, TreeNode<State, Action> child) {
			if (child.done) {
				return -std::numeric_limits<double>::infinity();
			}
			unsigned int total_visits = parent.visits;
			unsigned int visits = child.visits;
			double value = child.sum;
			double sigma = std::sqrt(child.stddev / visits);
            double maxChildUCB = child.state.maxChildUCB;
			return std::max(maxChildUCB, value/visits + this->hyperparameters.c * std::sqrt(std::log(total_visits) / visits) + child.state.getWeight());
		}


};





typedef MCTSUCB1Max<SampleGridState, Action> MCTSUCB1MaxGrid;
typedef MCTSUCB1Max<LeftDeterministicGridState, Action> MCTSUCB1MaxGridLDet;
typedef MCTSUCB1Max<RightDeterministicGridState, Action> MCTSUCB1MaxGridRDet;

typedef MCTSUCB1Avg<SampleGridState, Action> MCTSUCB1AvgGrid;
typedef MCTSUCB1Avg<LeftDeterministicGridState, Action> MCTSUCB1AvgGridLDet;
typedef MCTSUCB1Avg<RightDeterministicGridState, Action> MCTSUCB1AvgGridRDet;


typedef TreeMCTSUCB1Max<RightDeterministicGridState, Action> TreeMCTSUCB1MaxGridRDet;
typedef TreeMCTSUCB1AMax<RightDeterministicGridState, Action> TreeMCTSUCB1AMaxGridRDet;
typedef TreeMCTSUCB1Avg<RightDeterministicGridState, Action> TreeMCTSUCB1AvgGridRDet;
typedef TreeMCTSUCB1Sum<RightDeterministicGridState, Action> TreeMCTSUCB1SumGridRDet;
typedef TreeMCTSUCBMin<RightDeterministicGridState, Action> TreeMCTSUCBMinGridRDet;
typedef TreeMCTSBayes<RightDeterministicGridState, Action> TreeMCTSBayesGridRDet;

typedef TreeMCTSUCB1Max<SampleGridState, Action> TreeMCTSUCB1MaxGrid;
typedef TreeMCTSUCB1AMax<SampleGridState, Action> TreeMCTSUCB1AMaxGrid;
typedef TreeMCTSUCB1Avg<SampleGridState, Action> TreeMCTSUCB1AvgGrid;
typedef TreeMCTSUCB1Sum<SampleGridState, Action> TreeMCTSUCB1SumGrid;
typedef TreeMCTSUCBMin<SampleGridState, Action> TreeMCTSUCBMinGrid;
typedef TreeMCTSBayes<SampleGridState, Action> TreeMCTSBayesGrid;
typedef TreeMCTSMyMax<SampleGridState, Action> TreeMCTSMyMaxGrid;
typedef TreeMCTSGreedyUCB1Avg<SampleGridState, Action> TreeMCTSGreedyUCB1AvgGrid;
typedef TreeMCTSGreedyUCBMin<SampleGridState, Action> TreeMCTSGreedyUCBMinGrid;
typedef TreeMCTSGreedyBayes<SampleGridState, Action> TreeMCTSGreedyBayesGrid;

typedef TreeMCTSUCB1Max<SampleSmoothGridState, Action> TreeMCTSUCB1MaxSmoothGrid;
typedef TreeMCTSUCB1AMax<SampleSmoothGridState, Action> TreeMCTSUCB1AMaxSmoothGrid;
typedef TreeMCTSUCB1Avg<SampleSmoothGridState, Action> TreeMCTSUCB1AvgSmoothGrid;
typedef TreeMCTSUCB1Sum<SampleSmoothGridState, Action> TreeMCTSUCB1SumSmoothGrid;
typedef TreeMCTSUCBMin<SampleSmoothGridState, Action> TreeMCTSUCBMinSmoothGrid;
typedef TreeMCTSBayes<SampleSmoothGridState, Action> TreeMCTSBayesSmoothGrid;
typedef TreeMCTSMyMax<SampleSmoothGridState, Action> TreeMCTSMyMaxSmoothGrid;
typedef TreeMCTSGreedyUCB1Avg<SampleSmoothGridState, Action> TreeMCTSGreedyUCB1AvgSmoothGrid;
typedef TreeMCTSGreedyUCBMin<SampleSmoothGridState, Action> TreeMCTSGreedyUCBMinSmoothGrid;
typedef TreeMCTSGreedyBayes<SampleSmoothGridState, Action> TreeMCTSGreedyBayesSmoothGrid;
typedef BAST<SampleSmoothGridState, Action> BASTSmoothGrid;

typedef TreeMCTSUCBMin< GridStateImprovedSample, Action> TreeMCTSUCBMinImprovedSampleGrid;

typedef TreeMCTSGreedyUCB1Avg< GridStateImprovedSplit, Action> TreeMCTSGreedyUCB1AvgImprovedSplitGrid;
typedef TreeMCTSUCBMin< GridStateImprovedSplit, Action> TreeMCTSUCBMinImprovedSplitGrid;
typedef TreeMCTSGreedyBayes< GridStateImprovedSplit, Action> TreeMCTSGreedyBayesImprovedSplitGrid;


typedef TreeMCTSUCB1Max<FixPointGridState, Action> TreeMCTSUCB1MaxFixPointGrid;
typedef TreeMCTSUCB1AMax<FixPointGridState, Action> TreeMCTSUCB1AMaxFixPointGrid;
typedef TreeMCTSUCB1Avg<FixPointGridState, Action> TreeMCTSUCB1AvgFixPointGrid;
typedef TreeMCTSUCB1Sum<FixPointGridState, Action> TreeMCTSUCB1SumFixPointGrid;
typedef TreeMCTSUCBMin<FixPointGridState, Action> TreeMCTSUCBMinFixPointGrid;
typedef TreeMCTSBayes<FixPointGridState, Action> TreeMCTSBayesFixPointGrid;


#endif // MCTS_HPP
