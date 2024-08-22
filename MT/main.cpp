#include "mcts.h"
#include "hyperparams.h"
#include "point_set.h"
#include <chrono>
#include <fstream> // Include the necessary header for std::ofstream

int main() {
	const unsigned n = 500; // Number of points
	const unsigned d = 10;   // Dimension of the points
	// Target value:  0.1702
	// Best own value: 0.155384 (SampleGridState, TreeMCTSBayesGrid, mt(3), c = 0.01, its = 30000)
	// Through HP Search: Min, Max value: 0.162335, C: 0.00341095; Algorithm: Bayes, Max value : 0.162335, C : 1.16346 (5000 its)

	unsigned int its = 100;
	unsigned int multisample = 100;
	bool output = true;

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();

# pragma region BigEndComparison
	// Get all the files in the "PointSets" directory
	std::vector<std::string> filenames = {
		"Faure_50_10.txt",
		"Faure_100_10.txt",
		"Faure_121_8.txt",
		"Faure_121_9.txt",
		"Faure_121_10.txt",
		"Faure_121_11.txt",
		"Faure_169_12.txt",
		"Faure_343_7.txt",
		"Faure_500_10.txt",
		"Faure_529_20.txt",
		"Faure_1500_20.txt",
		"Faure_2000_50.txt",
		"Faure_4000_50.txt",
		"Halton_30_10.txt",
		"Halton_50_3.txt",
		"Halton_50_4.txt",
		"Halton_50_5.txt",
		"Halton_50_6.txt",
		"Halton_50_7.txt",
		"Halton_50_8.txt",
		"Halton_50_10.txt",
		"Halton_100_7.txt",
		"Halton_100_10.txt",
		"Halton_500_10.txt",
		"Halton_1000_7.txt",
		"Sobol_20_5.txt",
		"Sobol_50_5.txt",
		"Sobol_100_4.txt",
		"Sobol_100_5.txt",
		"Sobol_100_6.txt",
		"Sobol_100_8.txt",
		"Sobol_100_20.txt",
		"Sobol_100_50.txt",
		"Sobol_128_8.txt",
		"Sobol_128_9.txt",
		"Sobol_128_10.txt",
		"Sobol_128_11.txt",
		"Sobol_128_12.txt",
		"Sobol_128_20.txt",
		"Sobol_256_7.txt",
		"Sobol_256_12.txt",
		"Sobol_256_20.txt",
		"Sobol_500_5.txt",
		"Sobol_512_7.txt",
		"Sobol_1024_20.txt",
		"Sobol_2000_50.txt",
		"Sobol_2048_20.txt",
		"Sobol_4000_50.txt"
	};
	unsigned int small_its = 10000;
	unsigned int big_its = 100000;
	HPStatistic bigStatistic;
	for (const auto& filename : filenames) {
		AnonymousPointSet readSet = readFromFile("PointSets/" + filename);
		auto params = cLogEquidistant(1e-5, 10, 19);
		HPStatistic statistic;
		for (auto& p : params){
			statistic.addSingle(
				treeSingleSearchAnonymous<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(p, small_its, multisample, readSet, "G")
			);
		}
		// Select the hyperparameter with the best mean value
		double bestMean = -std::numeric_limits<double>::infinity();
        UCBHyperparameters bestParam;
        for (const auto& p : params) {
            if (statistic.get_average("G", p) > bestMean) {
                bestMean = statistic.get_average("G", p);
                bestParam = p;
            }
        }
		// Now perform the algorithm on best parameter
		std::string name = filename + "_G_" + std::to_string(bestParam.c) ; 
		bigStatistic.addSingle(
            treeSingleSearchAnonymous<SampleGridState, TreeMCTSUCB1Avg<SampleGridState, Action>>(bestParam, big_its, multisample, readSet, name)
        );
		std::cout << "Finished " << name << std::endl;
    }
	// Output the results
	bigStatistic.output_to_file("outputs/EndComparison_G.txt");


# pragma endregion BigEndComparison



# pragma region PointSetOutput
	// HaltonPointSet pointSet(d, n);
	// pointSet.generate();
	// std::string filename = "PointSets/Halton_" + std::to_string(n) + "_" + std::to_string(d) + ".txt";
	// pointSet.writeToFile(filename);

	// AnonymousPointSet readSet = readFromFile(filename);
	// // Do a check if the point sets are the same (up to precision 5 decimal places)
	// bool same = true;
    // for (unsigned i = 0; i < n; ++i) {
    //     for (unsigned j = 0; j < d; ++j) {
    //         if (std::abs(readSet.points[i][j] - pointSet.points[i][j]) > 1e-5) {
    //             same = false;
    //             break;
    //         }
    //     }
    //     if (!same) {
    //         break;
    //     }
    // }
    // if (output) {
    //     std::cout << "Point sets are the same: " << same << std::endl;
    // }
# pragma endregion PointSetOutput


# pragma endregion PointSetOutput


# pragma region StateComparison
# pragma region OutputAllStatesUCB1
	// auto params = cLogEquidistant(1e-5, 10, 19);
	// HPStatistic stat =  allStatesUCTStatistic(params, its, multisample, n, d);
	// stat.output_to_file("outputs/all_states_100_000_500_10.txt");
# pragma endregion OutputAllStatesUCB1


# pragma region OutputAllStatesSobol
	// auto params = cLogEquidistant(1e-5, 10, 6);
	// HPStatistic stat =  allStatesUCTStatisticSobol(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion OutputAllStatesSobol


# pragma region OutputAllStatesSobol
	// auto params = cLogEquidistant(1e-5, 10, 19);
	// HPStatistic stat =  PolicyCompare(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion OutputAllStatesSobol


# pragma region ValueTransforms
	// auto params = cLogEquidistant(1e-5, 10, 19);
	// HPStatistic stat =  ValueTransforms(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion ValueTransforms


# pragma region Restarts
	// auto params = cLogEquidistant(1e-5, 10, 19);
	// HPStatistic stat =  ValueTransforms(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion Restarts



# pragma region GridvsImproved // THis is not interesting
	// auto params = cLogEquidistant(0.000464159, 1e-2, 11);
	// HPStatistic stat = GridvsImprovedUCTStatistic(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion GridvsImproved

# pragma region TimelineImproved
	// UCBHyperparameters params(0.0001);
	// its = 1000000;
	// unsigned int steps = 100;
	// TimeLineStatistic stat = ImprovedSplitTimeline(params, its, steps, multisample, n, d);
	// stat.output_to_file("outputs/timeline_improved_0_0001.txt");
# pragma endregion TimelineImproved

# pragma region TimelineExploitation
	// UCBHyperparameters params(0.0001);
	// its = 10000;
	// unsigned int steps = 100;
	// TimeLineStatistic stat = ImprovedSplitTimeline(params, its, steps, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion TimelineExploitation

# pragma region TimelineExploration
	// UCBHyperparameters params(0.1);
	// its = 10000;
	// unsigned int steps = 100;
	// TimeLineStatistic stat = ImprovedSplitTimeline(params, its, steps, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion TimelineExploration

# pragma region TimelineLong
	// UCBHyperparameters params(0.001);
	// its = 100000;
	// unsigned int steps = 100;
	// TimeLineStatistic stat = ImprovedSplitTimeline(params, its, steps, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion TimelineLong

# pragma region ImprovedPointOutput
	// // // Generate Faure points
	// FaurePointSet pointSet(d, n);
	// pointSet.generate();
	// Grid grid(pointSet);
	// RightDeterministicGridState gridState(&grid);
	// std::mt19937 mt(1);
	// UCBHyperparameters params(0.0464159);
	// TreeMCTSUCB1Avg<RightDeterministicGridState, Action> mctsgrid(&pointSet, gridState, its * d * 3, mt, params);
	// mctsgrid.point_output = true;
	// mctsgrid.run(its);
	// // Print the max value
	// std::cout << mctsgrid.maxValue() << std::endl;
	// auto points_and_values = mctsgrid.points_and_values;
	// // Output to file
	// pointsAndValueToFile("outputs/debugPointOutput.txt", points_and_values);
# pragma endregion ImprovedPointOutput

# pragma region BigPointComparison
	// std::pair<unsigned int, unsigned int> nd = 
	// 	{ 
	// 		// 343, 7
	// 		// 121, 11
	// 		529, 20
	// 		// 1500, 20
	// 	};
	// its = 10000;
	// auto params = cLogEquidistant(1e-7, 1e-2, 16);
	// HPStatistic stat =  allStatesUCTStatistic(params, its, multisample, nd.first, nd.second);
	// stat.output_to_file("outputs/ucb1_" + std::to_string(nd.first) + "_" + std::to_string(nd.second) + "_" + std::to_string(its) + ".txt");
	// // Print n and d
	// std::cout << "n: " << nd.first << ", d: " << nd.second << " done" << std::endl;
# pragma endregion BigPointComparison





# pragma region CustomDiscrepancy
	// unsigned int dd = 11;
	// unsigned int nn = 121;
	// FaurePointSet pointSet(d, n);
	// pointSet.generate();
	// Grid grid(pointSet);
	// std::vector<double> point = {
    //     0.46938775510204078,
    //     0.98542274052478129,
    //     0.98250728862973757,
    //     0.98542274052478129,
    //     0.98250728862973757,
    //     0.98542274052478129,
    //     0.98542274052478129,
    //     0.46938775510204078,
    //     0.98542274052478129,
    //     0.98250728862973757,
    //     0.98542274052478129
    // };
	// // calculate the discrepancy of the point
	// double discrepancy = pointSet.discrepancy_snapped(point, grid);
	// std::cout << "Discrepancy: " << discrepancy << std::endl;
	// // Calculate point up snapped point
	// std::vector<double> up_snapped_point(d);
    // for (unsigned int i = 0; i < d; i++) {
    //     auto it = std::upper_bound(grid.grid[i].begin(), grid.grid[i].end(), point[i]);
    //     if (it == grid.grid[i].end()) {
    //         up_snapped_point[i] = 1;
    //     }
    //     else {
    //         up_snapped_point[i] = *it;
    //     }
    // }
	// std::cout << "Up snapped point: " << std::endl;
	// for (auto& p : up_snapped_point) {
	// 	std::cout << p << ", ";
	// }
	// std::cout << std::endl;
	// // Discrepancy of up snapped point
	// double discrepancy_up = pointSet.discrepancy(up_snapped_point);
	// std::cout << "Discrepancy up: " << discrepancy_up << std::endl;

	// // Calculate point down snapped point
	// std::vector<double> down_snapped_point(d);
	// for (unsigned int i = 0; i < d; i++) {
	// 	auto it = std::lower_bound(grid.grid[i].begin(), grid.grid[i].end(), point[i]);
	// 	if (it == grid.grid[i].begin()) {
	// 		down_snapped_point[i] = 0;
	// 	}
	// 	else {
	// 		down_snapped_point[i] = *(--it);
	// 	}
	// }
	// std::cout << "Down snapped point: " << std::endl;
	// for (auto& p : down_snapped_point) {
	// 	std::cout << p << ", ";
	// }
	// std::cout << std::endl;
	// // Discrepancy of down snapped point
	// double discrepancy_down = pointSet.discrepancy_bar(down_snapped_point);
	// std::cout << "Discrepancy down: " << discrepancy_down << std::endl;


	// // Print first coordinate of grid
	// std::cout << "First coordinate of grid: " << std::endl;
	// for (auto& p : grid.grid[0]) {
	// 	std::cout << p << ", ";
	// }
	// std::cout << std::endl;


# pragma endregion CustomDiscrepancy



# pragma endregion StateComparison

# pragma region PolicyComparison

# pragma region GridStatePolicyComparison
	// auto params = cLogEquidistant(1e-5, 1, 16);
	// HPStatistic stat =  GridPolicyCompare(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion GridStatePolicyComparison

# pragma region ImprovedStatePolicyComparison
	// auto params = cLogEquidistant(1e-5, 1, 6);
	// HPStatistic stat =  ImprovedSplitPolicyCompare(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion ImprovedPolicyComparison

# pragma endregion PolicyComparison


# pragma region RestartTest
	// // Generate Faure points
	// FaurePointSet pointSet(d, n);
	// pointSet.generate();
	// // Gridstate
	// Grid grid(pointSet);
	// SampleGridState gridState(&grid);
	// std::mt19937 mt(2);
	// UCBHyperparameters params(0.1);
	// typedef TreeMCTSUCB1Avg<SampleGridState, Action> UCBClass;
	// UCBClass tree = treeSearchwithRestarts<SampleGridState, UCBClass>(&pointSet,gridState, params, its);
	// // Print the max value
	// std::cout << tree.maxValue() << std::endl;
# pragma endregion RestartTest







# pragma region Experimenting

# pragma region HPSearch
	// auto params = cLogEquidistant(0.00001, 1, 10);
	// HPStatistic stat = gridStateAllStatistic(params, its, multisample, n, d);
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion HPSearch

# pragma region HPSearchOne
	// auto params = cLogEquidistant(1e-4, 1e-1, 10);
	// HPStatistic stat = treeOneSearchGrid<GridStateImprovedSplit, TreeMCTSGreedyBayesImprovedSplitGrid>(params, its, multisample, n, d, "GreedyAvgSplit");
	// stat.output_to_file("outputs/tmp.txt");
# pragma endregion HPSearchOne

#pragma region PointOutput
	//// Generate Faure points
	//FaurePointSet pointSet(d, n);
	//pointSet.generate();
	//// Gridstate
	//Grid grid(pointSet);
	//GridState gridState(&grid);
	//std::mt19937 mt(2);
	//UCBHyperparameters params(0.1);
	//TreeMCTSMyMaxGrid mctsgrid(&pointSet, gridState, its * d * 3, mt, params);
	//mctsgrid.point_output = true;
	//mctsgrid.run(its);
	//auto points_and_values = mctsgrid.points_and_values;
	//// Output to file
	//pointsAndValueToFile("../outputs/pointOutput.txt", points_and_values);
#pragma endregion PointOutput

	//Grid grid(pointSet);
	//SampleGridState gridState(&grid);
	//std::mt19937 mt(3);
	//UCBHyperparameters params(0.1);
	//TreeMCTSBayesGrid mctsgrid(&pointSet, gridState, its * d * 3, mt, params);
	//mctsgrid.run(its);
	//std::cout << mctsgrid.maxValue() << std::endl;

# pragma endregion Experimenting
	
	
	// End timer and print time
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	// Add at the end of tmp.txt
	std::ofstream file("outputs/tmp.txt", std::ios_base::app);
	if (file.is_open()) { // Check if the file is successfully opened
		file << "Time: " << elapsed.count() << "s" << std::endl;
		file.close();
	} else {
		std::cerr << "Failed to open the file." << std::endl; // Print an error message if the file failed to open
	}
	return 0;
}
