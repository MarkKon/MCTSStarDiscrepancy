#ifndef POINT_SET_HPP
#define POINT_SET_HPP

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include "faure.hpp"

class Grid;

class PointSet {
public:
    unsigned int d;
    unsigned int n;

    std::vector<std::vector<double>> points;

    PointSet() {
		this->d = 0;
        this->n = 0;
	}

    PointSet(unsigned int d, unsigned int n) {
        this->d = d;
		this->n = n;
    }

    PointSet(const PointSet& p) {
        this->d = p.d;
        this->n = p.n;
    }

    PointSet& operator=(const PointSet& p) {
        return *this;
    }

    virtual void generate() = 0;

    double discrepancy(const std::vector<double>& point);

    double discrepancy_bar(const std::vector<double>& point);

    // snapped discrepancy with a given grid (see class Grid)
    double discrepancy_snapped(const std::vector<double>& point, const Grid& grid);

    void writeToFile(const std::string& filename) {
        std::ofstream file;
		file.open(filename);
		for (unsigned int i = 0; i < n; i++) {
			for (unsigned int j = 0; j < d; j++) {
				file << points[i][j] << ",";
			}
			file << std::endl;
		}
		file.close();
    }
};


class Grid {
public:
    // Grid projected by the points. The grid is a d-dimensional vector of vectors of doubles (the projections of the points, sorted, 0 and 1 included)
    std::vector<std::vector<double>> grid;

    // Constructors
    // Empty grid
    Grid() {
        grid = std::vector<std::vector<double>>(0);
    };
    // Grid from a point set
    Grid(const PointSet& p);

    // Compute the number of grid points per dimension
    std::vector<unsigned int> gridDims();

};


// Class for a point set that has been anonymized i.e. just stores points without any other information
class AnonymousPointSet : public PointSet {
public:
    AnonymousPointSet(unsigned int d, unsigned int n) : PointSet(d, n) {};
    // directly use points ie. use std::vector<std::vector<double>> as input
    AnonymousPointSet(const std::vector<std::vector<double>>& points) : PointSet(points[0].size(), points.size()) {
        // Check that all points have the same dimension
        for (unsigned int i = 0; i < points.size(); i++) {
            if (points[i].size() != d) {
				throw std::invalid_argument("All points must have the same dimension");
			}
		}
        // Check that all points are in the unit cube
        for (unsigned int i = 0; i < points.size(); i++) {
            for (unsigned int j = 0; j < d; j++) {
                if (points[i][j] < 0 || points[i][j] > 1) {
                    throw std::invalid_argument("All points must be in the unit cube");
                }
            }
        }

		this->points = points;
	}
    void generate(){};
};

class UniformRandomPointSet : public PointSet {
public:

    UniformRandomPointSet(unsigned int d, unsigned int n) : PointSet(d, n) {};
    void generate();
};

class FaurePointSet : public PointSet {
public:
    FaurePointSet(unsigned int d, unsigned int n) : PointSet(d, n) {};
	void generate();
};

#endif // POINT_SET_HPP
