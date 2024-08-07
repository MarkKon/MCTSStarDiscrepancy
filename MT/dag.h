#pragma once
// Abstraction of a directed acyclic graph to be used for the mcts algorithm
// Put a template over the State type and the action type
#include <unordered_map>
#include <map>

template <class State, class Action>
struct Package
{
	unsigned int visits;
	double sum;
	double max;
	std::vector<std::pair<Action,State>> actions;
	bool leaf;
	bool done;
};

// Any State class must have the following methods:
//		- getActions() : returns a vector of all the actions that can be taken from this state
//		- isLeaf() : returns a boolean indicating if the state is a leaf node or not
//		- operator+ : returns a new state that is the result of the action on the state
//		- operator< : For the unordered_map to work, we need to define a comparison operator
//		- sample() : returns a point from the state space


template <class State, class Action>
class DAG
{
	// At the heart of the Dag is a map from states to all the additional information linked to them
	// We want to encompass the following information:
	// 	   - Numerical information about the state
	//		 - The number of visits
	//		 - The sum of the values of the visits (for the average)
	//		 - The maximum value of the visits
	//	   - Information the children
	//		 - Is it a leaf node or not? -> Note that this can differ from the state being a leaf (think of pruning)
	//       - List of actions that can be taken from this node
	// 	       - Values of the actions (number of visits, sum of values, maximum value) for quick access
	//       - List of children (as refs)
	//	   - Information on parents
	//		 - List of direct parents

	// We will use an unordered map as it is the most efficient way to store the information
		// We will use a pointer to the state as the key, and a struct as the value
		// The struct will contain the numerical information and the children
		// The children will be stored as a vector of pointers to the states
		// The struct will also contain a vector of pointers to the parents
		// The struct will also contain a vector of actions that can be taken from this state
		// The actions will be stored as a vector of structs containing the numerical information of the actions

public:
	State root;
	std::map<State, Package<State, Action>> states;
			
	DAG()
	{
		states = std::map<State, Package<State, Action>>();
		root = State();
	}

	DAG(State root, const unsigned int allocation_size)
	{
		states = std::map<State, Package<State, Action>>();
		// states.reserve(allocation_size);
		this->root = root;
		// Add root to the map
		this->addState(root);
	}

	DAG(DAG& dag)
	{
		root = dag.root;
		states = dag.states;
	}

	void addState(State state)
	{
		if (states.find(state) != states.end())
		{
			return;
		}
		// Calculate the children
		std::vector<Action> actions = state.actions();
		std::vector<State> children;
		for (auto action : actions)
		{
			State child = state + action;
			children.push_back(child);
		}
		// Construct action/state pairs
		std::vector<std::pair<Action, State>> action_state_pairs;
		for (unsigned int i = 0; i < actions.size(); i++)
		{
			action_state_pairs.push_back(std::make_pair(actions[i], children[i]));
		}

		// Add the state to the map
		states[state] = { 0, 0, 0, action_state_pairs, state.isLeaf(), false};
	}

	void addVisit(State state, double value)
	{
		// Get the package
		Package<State, Action>& package = states[state];
		// Update the package
		package.visits++;
		package.sum += value;
		if (value > package.max)
			package.max = value;

	}

	unsigned int getVisits(State state)
	{
		// Check if the state is in the map
		if (states.find(state) == states.end())
			return 0;
		return states[state].visits;
	}

	double getSum(State state)
	{
		if (states.find(state) == states.end())
			return 0;
		return states[state].sum;
	}

	double getMax(State state)
	{
		if (states.find(state) == states.end())
			return 0;
		return states[state].max;
	}


	Package<State, Action> getPackage(State state)
	{
		if (states.find(state) == states.end())
			throw std::invalid_argument("State not in the map");
		return states[state];
	}

	std::vector<State> getChildren(State state)
	{
		if (states.find(state) == states.end())
			return std::vector<State>();
		std::vector<State> children;
		for (auto& action_state : states[state].actions)
		{
			children.push_back(action_state.second);
		}
		return children;
	}

	std::vector<Action> getActions(State state)
	{
		if (states.find(state) == states.end())
			return std::vector<Action>();
		std::vector<Action> actions;
		for (auto& action_state : states[state].actions)
		{
			actions.push_back(action_state.first);
		}
		return actions;
	}

	void removeAction(State state, Action action)
	{
		if (states.find(state) == states.end())
			return;
		// Remove the action from the list of actions_states
		auto& actions_states = states[state].actions;
		actions_states.erase(std::remove_if(actions_states.begin(), actions_states.end(), 
			[action](std::pair<Action, State> action_state) 
				{
					return action_state.first == action; 
				}), 
			actions_states.end());
		// If the state is now a leaf, update the leaf status
		if (actions_states.size() == 0) {
			states[state].leaf = true;
			states[state].done = true;
		}
	}

	bool isLeaf(State state)
	{
		if (states.find(state) == states.end())
			return state.isLeaf();
		return states[state].leaf;
	}

	void setLeaf(State state, bool leaf)
	{
		if (states.find(state) == states.end())
			return;
		states[state].leaf = leaf;
	}

	bool isDone(State state)
	{
		if (states.find(state) == states.end())
			return false;
		return states[state].done;
	}

	void setDone(State state)
	{
		if (states.find(state) == states.end())
			return;
		states[state].done = true;
	}

	// Depth statistics: Vector of number of states at each depth
	// Depth for each state can be calculated by state.depth.
	std::vector<unsigned int> depthStatistics()
	{
		std::vector<unsigned int> depth_stats;
		for (auto& state : states)
		{
			unsigned int depth = state.first.depth;
			if (depth_stats.size() <= depth)
				depth_stats.resize(depth + 1);
			depth_stats[depth]++;
		}
		return depth_stats;
	}

};

template <class State, class Action>
struct TreeNode
{
	unsigned int visits;
	double sum;
	double max;
	double stddev;
	State state;
	std::vector<std::pair<Action, TreeNode*>> actions;
	bool leaf;
	bool done;

	TreeNode()
	{
		visits = 0;
		sum = 0;
		max = 0;
		stddev = 0;
		state = State();
		actions = std::vector<std::pair<Action, TreeNode*>>();
		leaf = state.isLeaf();
		done = false;
	}

	TreeNode(State state)
	{
		visits = 0;
		sum = 0;
		max = 0;
		stddev = 0;
		this->state = state;
		actions = std::vector<std::pair<Action, TreeNode*>>();
		leaf = state.isLeaf();
		done = false;
	}

	TreeNode<State, Action>& operator=(const TreeNode& node)
	{
		visits = node.visits;
		sum = node.sum;
		max = node.max;
		stddev = node.stddev;
		state = node.state;
		actions = node.actions;
		leaf = node.leaf;
		done = node.done;
		return *this;
	}

	TreeNode* getChild(Action action)
	{
		for (auto& action_node : actions)
		{
			if (action_node.first == action)
				return action_node.second;
		}
		return nullptr;
	}

	void addChild(Action action, TreeNode* child)
	{
		actions.push_back(std::make_pair(action, child));
	}

	void removeChild(Action action)
	{
		actions.erase(std::remove_if(actions.begin(), actions.end(), 
			[action](std::pair<Action, TreeNode*> action_node)
			{
					return action_node.first == action; 
				}), 
						actions.end());
	}
};

template<class State, class Action>
struct Path
{
	std::vector<std::pair<State, Action>> path;

	Path()
	{
		path = std::vector<std::pair<State, Action>>();
	}

	Path(unsigned int size)
	{
		path = std::vector<std::pair<State*, Action>>();
		path.reserve(size);
	}

	void push_back(State state, Action action)
	{
		path.push_back(std::make_pair(state, action));
	}

	std::pair<State, Action> back()
	{
		return path.back();
	}

	State back_state()
	{
		return path.back().first;
	}

	Action back_action()
	{
		return path.back().second;
	}

	void pop_back()
	{
		path.pop_back();
	}

	// Get the size of the path
	unsigned int size()
	{
		return path.size();
	}

};

template<class State, class Action>
struct NodePath
{
	// A NodePath is a list of Node pointer, Action pairs
	std::vector<std::pair<TreeNode<State, Action>*, Action>> path;
	NodePath()
	{
		path = std::vector<std::pair<TreeNode<State, Action>*, Action>>();
	}
	void push_back(TreeNode<State, Action>* node, Action action)
	{
		path.push_back(std::make_pair(node, action));
	}
	std::pair<TreeNode<State, Action>*, Action> back()
	{
		return path.back();
	}
	TreeNode<State, Action>* back_node()
	{
		return path.back().first;
	}
	Action back_action()
	{
		return path.back().second;
	}
	void pop_back()
	{
		path.pop_back();
	}
	unsigned int size()
	{
		return path.size();
	}
	// define indexing operator
	std::pair<TreeNode<State, Action>*, Action>& operator[](unsigned int index)
	{
		return path[index];
	}
};