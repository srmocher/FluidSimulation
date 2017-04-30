#pragma once
#ifndef  MARCHING_CUBES
#define MARCHING_CUBES
#endif // ! MARCHING_CUBES

#include <glm/glm.hpp>
#include <vector>
#include "Headers.h"
class Cell
{
public:
	float x, y, z;
};

class Cube {
public:
	glm::vec3 pos[8];
	int value;
	float values[8];
};
class MarchingCubes
{
private:
	glm::vec3 *particlePosition;
	float mass;
	float *particleDensity;
	particleData pData;
	int particleCount;
public:
	MarchingCubes(glm::vec3 *,float *,float,particleData,int);
	int run(int i,int j,int k, Cube ***cube,int *cubeMap, glm::vec3 pos,float,float,float *vals);
	glm::vec3 interpolate(glm::vec3 p1, glm::vec3 p2,float isoValue,float val1,float val2);
	double getColorField(glm::vec3 point);
};

