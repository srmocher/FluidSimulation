#pragma once
#ifndef __SPH_CONTAINER__
#define __SPH_CONTAINER__


#include "sph.cuh"
#include <glm/glm.hpp>
#include "MarchingCubes.h"

class Cube;
struct particleInfo
{
    unsigned int particleCount;
    float activeRadius;
    float fluidDensity;
    float fluidViscosity;
    float stiffness; 
    

    float maximumSpeed;
    float maximumAcceleration;


};

class SphContainer
{
public:
    SphContainer(float x, float y, float z, float w, float l, float h);
    void drawContainer();
    void createParticles(particleInfo pInfo);
    void drawParticles();

    void sendDataToGPU();
    void getDataFromGPU();
    void computeFluid(float dt);
    void setPower(float power, float rad, glm::vec3 pos, glm::vec3 vel);
	void drawParticle(float x,float y,float z);
	bool checkParticleExists(float x, float y, float z,float, float&,int,int,int );
	void calculateDensities(int i,int j, int k);
	double getColorFieldValue(glm::vec3 point);
		~SphContainer(void);;
		void runMC();

private:

	Cube ***cube;
    unsigned int particleCount;

    particleData pData;

    float viscosity;
	bool initial = true;
    float mass;
    float rest_pressure;
    float rest_density;
    float radius;
    float pressure_koeff;

    float r2;
    float r3;


    glm::vec3* particlePosition;
    glm::vec4* particleVelocity;
    glm::vec3* particleHvelocity;
    float* particleDensity;
    glm::vec3* particleColor;
	float* particleViscosity;
    int* particleIndex;
    int* particleZindex;
    
    int* particleBeg;
    int* particleEnd;
	float *colorField;

    cudaGraphicsResource* cudaPosVbo;
    cudaGraphicsResource* cudaColorResource;

    


    unsigned int particlePositionVBO1;
    unsigned int particleColorVBO;
    unsigned int particleMaxCount;
    

    float width;
    float height;
    float length;

    float centerX;
    float centerY;
    float centerZ;

    float divideStepSize;

    glm::vec3 deflectorPos;
    float deflectorRad;



    glm::vec3* containerDrawVertex;
    unsigned int* containerDrawIndex;
};



#endif