#include "point_set.h"

double PointSet::discrepancy(const std::vector<double>& point) {
    unsigned int numPointsInHyperrectangle = 0;
    double volumeOfHyperrectangle = 1.0;

    for (unsigned int i = 0; i < n; i++) {
        bool inHyperrectangle = true;
        for (unsigned int j = 0; j < d; j++) {
            if (points[i][j] >= point[j]) {
                inHyperrectangle = false;
                break;
            }
        }
        if (inHyperrectangle) {
            numPointsInHyperrectangle++;
        }
    }

    double relative_points = static_cast<double>(numPointsInHyperrectangle) / static_cast<double>(n);

    for (unsigned int i = 0; i < d; i++) {
        volumeOfHyperrectangle *= point[i];
    }

    return std::fabs(relative_points - volumeOfHyperrectangle);
}

double PointSet::discrepancy_bar(const std::vector<double>& point)
{
    unsigned int numPointsInHyperrectangle = 0;
	double volumeOfHyperrectangle = 1.0;

    for (unsigned int i = 0; i < n; i++) {
		bool inHyperrectangle = true;
        for (unsigned int j = 0; j < d; j++) {
            if (points[i][j] > point[j]) {
				inHyperrectangle = false;
				break;
			}
		}
        if (inHyperrectangle) {
			numPointsInHyperrectangle++;
		}
	}

	double relative_points = static_cast<double>(numPointsInHyperrectangle) / static_cast<double>(n);

    for (unsigned int i = 0; i < d; i++) {
		volumeOfHyperrectangle *= point[i];
	}

	return std::fabs(relative_points - volumeOfHyperrectangle); 
}

double PointSet::discrepancy_snapped(const std::vector<double>& point, const Grid& grid) {
    // snap the point up to the nearest upper grid point
    std::vector<double> up_snapped_point(d);
    for (unsigned int i = 0; i < d; i++) {
        auto it = std::upper_bound(grid.grid[i].begin(), grid.grid[i].end(), point[i]);
        if (it == grid.grid[i].end()) {
            up_snapped_point[i] = 1;
        }
        else {
            up_snapped_point[i] = *it;
        }
    }
    // snap the point down to the nearest lower grid point
    std::vector<double> down_snapped_point(d);
    for (unsigned int i = 0; i < d; i++) {
        auto it = std::lower_bound(grid.grid[i].begin(), grid.grid[i].end(), point[i]);
        if (it == grid.grid[i].begin()) {
            down_snapped_point[i] = 0;
        }
        else {
            down_snapped_point[i] = *(--it);

        }
    }
    // return max of d(y+), dbar(y-)
    return (double) std::max(discrepancy(up_snapped_point), discrepancy_bar(down_snapped_point));
}

// Read from file and return Anonymos Point Set


AnonymousPointSet readFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    unsigned int n, d;
    std::string line;

    // Read the first line to get n and d
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string temp;
        std::getline(ss, temp, ',');
        n = std::stoi(temp);
        std::getline(ss, temp, ',');
        d = std::stoi(temp);
    } else {
        throw std::runtime_error("File format error: Could not read n and d");
    }

    // Create the PointSet object
    AnonymousPointSet pointSet(d, n);
    pointSet.points.resize(n, std::vector<double>(d));

    // Read the points
    for (unsigned int i = 0; i < n; i++) {
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string temp;
            for (unsigned int j = 0; j < d; j++) {
                if (std::getline(ss, temp, ',')) {
                    pointSet.points[i][j] = std::stod(temp);
                } else {
                    throw std::runtime_error("File format error: Not enough points in line");
                }
            }
        } else {
            throw std::runtime_error("File format error: Not enough lines for points");
        }
    }

    file.close();
    return pointSet;
}


Grid::Grid(const PointSet& p) {
    // Create the grid
    grid = std::vector<std::vector<double>>(p.d);
    // For any dimension
    for (unsigned int i = 0; i < p.d; i++) {
        // Get all the projections of the points
        std::vector<double> projections(p.n);
        for (unsigned int j = 0; j < p.n; j++) {
            projections[j] = p.points[j][i];
        }
        // Add 0 and 1 to the projections
        projections.push_back(0);
        projections.push_back(1);
        // Sort the projections
        std::sort(projections.begin(), projections.end());
        // Remove duplicates
        projections.erase(std::unique(projections.begin(), projections.end()), projections.end());
        // Add the projections to the grid
        grid[i] = projections;
    }
};

std::vector<unsigned int> Grid::gridDims() {
    std::vector<unsigned int> gridPointsPerDimension(grid.size());
    for (unsigned int i = 0; i < grid.size(); i++) {
        gridPointsPerDimension[i] = grid[i].size();
    }
    return gridPointsPerDimension;
};

void UniformRandomPointSet::generate() {
    for (unsigned int i = 0; i < n; i++) {
        std::vector<double> point;
        for (unsigned int j = 0; j < d; j++) {
            point.push_back(static_cast<double>(rand()) / RAND_MAX);
        }
        this->points.push_back(point);
    }
}

void FaurePointSet::generate() {
	// use faure_generate to generate the points
    double* faure = faure_generate(d, n, 1);
    // this is array of size d*n
    for (unsigned int i = 0; i < n; i++) {
		std::vector<double> point;
        for (unsigned int j = 0; j < d; j++) {
			point.push_back(faure[i*d + j]);
		}
		this->points.push_back(point);
	}
}

void HaltonPointSet::generate() {
    double* halton = halton_sequence(1, n, d);
    for (unsigned int i = 0; i < n; i++) {
        std::vector<double> point;
        for (unsigned int j = 0; j < d; j++) {
            point.push_back(halton[i*d + j]);
        }
        this->points.push_back(point);
    }
}

void SobolPointSet::generate() {
    double* sobol = i8_sobol_generate(d, n, 1);
    for (unsigned int i = 0; i < n; i++) {
        std::vector<double> point;
        for (unsigned int j = 0; j < d; j++) {
            point.push_back(sobol[i*d + j]);
        }
        this->points.push_back(point);
    }
}