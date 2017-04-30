#pragma once
#include <vector>
class Cell;
class Grid
{
private:
	std::vector<Cell> cells;
	float step;
public:
	Grid(float s) { step = s; }
	void addCell(Cell c) {
		cells.push_back(c);
	}
};

class Cell
{
public:
	float x, y, z;

};