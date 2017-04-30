#include "SphContainer.h"
#include <time.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#ifndef MARCHING_CUBES
	#include "MarchingCubes.h"
#endif
float MPI = acos(-1.0);

SphContainer::SphContainer(float x, float y, float z, float w, float l, float h)
{    

    containerDrawVertex = new glm::vec3[24];
    containerDrawIndex = new unsigned int[24];

    float dx = x + w / 2;
    float dxm = x - w / 2;
    float dy = y + l / 2;
    float dym = y - l / 2;
    float dz = z + h / 2;
    float dzm = z - h / 2;

	//std::cout << dxm << "," << dym << "," << dzm<<std::endl;
    containerDrawVertex[0] = glm::vec3(dxm, dym, dzm);
//	std::cout << dx << "," << dym << "," << dzm<<std::endl;
    containerDrawVertex[1] = glm::vec3(dx, dym, dzm);
    containerDrawVertex[2] = glm::vec3(dx, dy, dzm);
    containerDrawVertex[3] = glm::vec3(dxm, dy, dzm);

    containerDrawVertex[4] = glm::vec3(dxm, dym, dz);
    containerDrawVertex[5] = glm::vec3(dxm, dy, dz);
    containerDrawVertex[6] = glm::vec3(dx, dy, dz);
    containerDrawVertex[7] = glm::vec3(dx, dym, dz);


    containerDrawVertex[8] = glm::vec3(dxm, dy, dzm);
    containerDrawVertex[9] = glm::vec3(dx, dy, dzm);
    containerDrawVertex[10] = glm::vec3(dx, dy, dz);
    containerDrawVertex[11] = glm::vec3(dxm, dy, dz);


    containerDrawVertex[12] = glm::vec3(dxm, dym, dzm);
    containerDrawVertex[13] = glm::vec3(dxm, dym, dz);
    containerDrawVertex[14] = glm::vec3(dx, dym, dz);
    containerDrawVertex[15] = glm::vec3(dx, dym, dzm);

    containerDrawVertex[16] = glm::vec3(dx, dym, dzm);
    containerDrawVertex[17] = glm::vec3(dx, dym, dz);
    containerDrawVertex[18] = glm::vec3(dx, dy, dz);
    containerDrawVertex[19] = glm::vec3(dx, dy, dzm);

    containerDrawVertex[20] = glm::vec3(dxm, dym, dzm);
    containerDrawVertex[21] = glm::vec3(dxm, dy, dzm);
    containerDrawVertex[22] = glm::vec3(dxm, dy, dz);
    containerDrawVertex[23] = glm::vec3(dxm, dym, dz);

    for (int i = 0; i < 24; ++i)
    {
        containerDrawIndex[i] = i;
    }
    centerX = x;
    centerY = y;
    centerZ = z;

    width = w;
    length = l;
    height = h;


}

void SphContainer::createParticles(particleInfo pInfo)
{

    particleCount = pInfo.particleCount;


    rest_density = pInfo.fluidDensity;
    viscosity = pInfo.fluidViscosity;
    radius = pInfo.activeRadius;
    pressure_koeff = pInfo.stiffness;
    r2 = radius * radius;
    r3 = radius * radius * radius;
    mass = 4.0f * r3 * rest_density * MPI / 3.0f / 19;

    float OFFSET = radius * 0.6;




    particlePosition = new glm::vec3[particleCount];
    particleVelocity = new glm::vec4[particleCount];
    particleHvelocity = new glm::vec3[particleCount];
    particleDensity  = new float[particleCount];
    particleColor = new glm::vec3[particleCount];
    particleIndex = new int[particleCount];
    particleZindex = new int[particleCount];
	particleViscosity = new float[particleCount];
	colorField = new float[particleCount];


    for (int i = 0; i < particleCount; ++i)
    {
        particleVelocity[i] = glm::vec4(0,0,0,0);
        particleHvelocity[i] = glm::vec3(0,0,0);

        particleDensity[i] = 0;
        particleIndex[i] = i;
        particleZindex[i] = 0;
        particleColor[i] = glm::vec3(1, 1, (rand() % 100) / 500  + 0.05);
		colorField[i] = 0.0f;
    }



	/*for (int i = 12001;i < particleCount;i++)
	{
		particleViscosity[i] = 8.5f;
	}*/

    glm::vec3 tempPos = glm::vec3(centerX - width / 2 + OFFSET, centerY - length / 2 + OFFSET,
        centerZ - height / 2 + OFFSET);
    glm::vec3 addPos = tempPos;
    int cnt = 0;
    while(cnt < particleCount)
    {
        for(int i = 0; i * i * i < particleCount && cnt < particleCount; ++i)
        {
            for(int j = 0; j * j * j < particleCount && cnt < particleCount; ++j)
            {
                particlePosition[cnt] = addPos + glm::vec3((rand() % 100) / 1000.0, (rand() % 100) / 1000.0, (rand() % 100) / 1000.0);
                addPos.x += OFFSET;
                ++cnt;
            }
            addPos.y += OFFSET;
            addPos.x = tempPos.x;
        }
        addPos.z += OFFSET;
        addPos.y = tempPos.y;
    }

	for (int i = 0;i < particleCount / 2;i++) {
		particleViscosity[i] = 2.5f;
	}

	for (int i = particleCount / 2;i < particleCount;i++) {
		particleViscosity[i] = 2.5f;
	}
	


    glGenBuffers(1, &particlePositionVBO1);
    glBindBuffer(GL_ARRAY_BUFFER, particlePositionVBO1);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(glm::vec3), particlePosition, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &particleColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleColorVBO);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(glm::vec3), particleColor, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    gpuErrchk( cudaGraphicsGLRegisterBuffer(&cudaPosVbo, particlePositionVBO1, cudaGraphicsMapFlagsWriteDiscard));
    gpuErrchk( cudaGraphicsGLRegisterBuffer(&cudaColorResource, particleColorVBO, cudaGraphicsMapFlagsWriteDiscard));

    const unsigned int HASH_TABLE_SIZE = 1453021;//prime number;

    particleBeg = new int[HASH_TABLE_SIZE];
    particleEnd = new int[HASH_TABLE_SIZE];


    // gpuErrchk( cudaMalloc(&particlePositionGPU, count * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.vel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.posTextured, particleCount * sizeof(float) * 4));
    gpuErrchk( cudaMalloc(&pData.accel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.hVel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.dens, particleCount * sizeof(float)));
    gpuErrchk(cudaMalloc(&pData.zind, particleCount * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.pind, particleCount * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.hashTableStart, HASH_TABLE_SIZE * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.hashTableEnd, HASH_TABLE_SIZE * sizeof(int)));

    gpuErrchk( cudaMalloc(&pData.tempVel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.temphVel, particleCount * sizeof(float) * 3));
	gpuErrchk(cudaMalloc(&pData.visc, particleCount * sizeof(float)));
	gpuErrchk(cudaMalloc(&pData.colorField, particleCount * sizeof(float)));



    // gpuErrchk(cudaMemcpy(particlePositionGPU, particlePosition, count * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.vel, particleVelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.hVel, particleHvelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.tempVel, particleVelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.temphVel, particleHvelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(pData.visc, particleViscosity, particleCount * sizeof(float), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(pData.colorField, colorField, particleCount * sizeof(float), cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMemcpy(pData.dens, particleDensity, particleCount * sizeof(float), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(pData.pind, particleIndex, particleCount * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(pData.zind, particleZindex, particleCount * sizeof(int), cudaMemcpyHostToDevice));


    pData.gravity.x = 0;
    pData.gravity.y = 0;
    pData.gravity.z = -9.8;

    pData.count = particleCount;
    pData.mass = mass;
    pData.r = radius;
    pData.r2 = r2;
    pData.rest_pressure = 0;
    pData.rest_density = rest_density;
    pData.viscosity = viscosity;
    pData.pressure_Koeff = pressure_koeff;
    pData.center.x = centerX;
    pData.center.y = centerY;
    pData.center.z = centerZ;
    pData.sizeContainer.x = width;
    pData.sizeContainer.y = length;
    pData.sizeContainer.z = height;
    pData.gridDimConst = radius * 1.2;

    pData.maxAcceleration = 1000;
    pData.wallDamping = 0.1;

    pData.diffKern = 315.0f / (64.0 * MPI * r3 * r3 * r3);
    pData.pressKern = 45.0 / (MPI * r3 * r3);
    pData.viscKern = 45.0 / (MPI * r3 * r3);

    pData.HASH_TABLE_SIZE = HASH_TABLE_SIZE;

    bindToTextures(&pData);
}

float lstTime = 1;
forceData frc;


void SphContainer::setPower(float power, float rad, glm::vec3 pos, glm::vec3 vel)
{
    frc.coord = make_float3(pos.x,  pos.y, pos.z);
    frc.velocity = make_float3(vel.x,  vel.y, vel.z);
    frc.radius = rad;
    frc.r2 = rad * rad;
    frc.power = power;
}

void SphContainer::computeFluid(float dt)
{
    gpuErrchk( cudaGraphicsMapResources(1, &cudaPosVbo, NULL));
    size_t size;
    gpuErrchk( cudaGraphicsResourceGetMappedPointer((void** )&pData.pos, &size, cudaPosVbo));        
    gpuErrchk( cudaGraphicsMapResources(1, &cudaColorResource, NULL));
    gpuErrchk( cudaGraphicsResourceGetMappedPointer((void** )&pData.color, &size, cudaColorResource));



    
        updateSimData(pData);
        prepareFluidGPU(pData, dt);
        updateSimData(pData);
        sph(pData, dt,frc);
		//gpuErrchk(cudaMemcpy(colorField, pData.colorField, particleCount * sizeof(float), cudaMemcpyDeviceToHost));
		
		/*for (int i = 0;i < particleCount;i++)
		{
			std::cout << colorField[i] << std::endl;
		}
*/
 //   cudaMemcpy(particleZindex, pData.zind, sizeof(int) * particleCount, cudaMemcpyDeviceToHost);
 //   cudaMemcpy(particleIndex, pData.pind, sizeof(int) * particleCount, cudaMemcpyDeviceToHost);
		
    gpuErrchk( cudaGraphicsUnmapResources(1, &cudaPosVbo, NULL));    
    gpuErrchk( cudaGraphicsUnmapResources(1, &cudaColorResource, NULL));
}






void SphContainer::drawParticles()
{     
	glPointSize(3);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, particlePositionVBO1);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, particleColorVBO);
    glColorPointer(3, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_POINTS, 0, pData.count);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);    
	/*gpuErrchk(cudaMemcpy(particlePosition, pData.pos, 3 * sizeof(float)*particleCount, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(particleDensity, pData.dens, sizeof(float)*particleCount, cudaMemcpyDeviceToHost));
	runMC();*/
}


void SphContainer::drawParticle(float x,float y,float z)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	//glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color4_sphere);

	glTranslatef(x, y,z);
	//glScalef(FRAME_SCALE[0], FRAME_SCALE[1], FRAME_SCALE[2]);	
	glRotated(0, 0, 0, 0);
	glutSolidSphere(0.03f, 100, 100);

	glPopMatrix();
}

void SphContainer::drawContainer()
{

    glm::vec3* p = containerDrawVertex;
    for (int i = 0; i < 6; ++i)
    {

        glBegin(GL_LINE_LOOP);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glEnd();    
    }


}


SphContainer::~SphContainer(void)
{
    delete[] containerDrawIndex;
    delete[] containerDrawVertex; 
}


void SphContainer::runMC()
{
	int cubeX = 64;
	int cubeY = 64;
	int cubeZ = 64;
	cube = new Cube**[cubeX];
	float x = centerX - width / 2, y = centerY - length / 2, z = centerZ - height / 2;
	float cubeSize = width / cubeX;

	for (int i = 0;i < cubeX;i++)
	{
		cube[i] = new Cube*[cubeY];
		
		y = centerY - length / 2;
		for (int j = 0;j < cubeY;j++)
		{
			cube[i][j] = new Cube[cubeZ];
			for (int k = 0;k < cubeZ;k++)
			{
				for (int l = 0;l < 8;l++)
					cube[i][j][k].values[l] = 0;
			}

		}
	}
	
	for (int i = 0;i < cubeX;i++)
	{
		
		
		y = centerY - length / 2;
		for (int j = 0;j < cubeY;j++)
		{
			
			z = centerZ - height / 2;
			for (int k = 0;k < cubeZ;k++)
			{
				float density = 0.0;
				bool particleExists = checkParticleExists(x, y, z,cubeSize,density,i,j,k);
				cube[i][j][k].pos[3] = glm::vec3(x, y, z);
				cube[i][j][k].pos[0] = glm::vec3(x, y, z + cubeSize);
				cube[i][j][k].pos[1] = glm::vec3(x + cubeSize, y, z + cubeSize);
				cube[i][j][k].pos[2] = glm::vec3(x + cubeSize, y, z);
				cube[i][j][k].pos[4] = glm::vec3(x, y + cubeSize, z + cubeSize);
				cube[i][j][k].pos[5] = glm::vec3(x + cubeSize, y + cubeSize, z + cubeSize);
				cube[i][j][k].pos[6] = glm::vec3(x + cubeSize, y + cubeSize, z);
				cube[i][j][k].pos[7] = glm::vec3(x, y + cubeSize, z);
				//double cfVal = getColorFieldValue(glm::glm::vec3(x, y, z));
				/*if (cfVal > 0.6) {
					cube[i][j][k].value = 1;
					
				}
				else {
					cube[i][j][k].value = 0;
				}*/
				if (particleExists) {
					cube[i][j][k].value = 1;
					calculateDensities(i, j, k);
				}
				else {
					cube[i][j][k].value = 0;
				}
				
				z = z + cubeSize;
			}
			y = y + cubeSize;
		}
		x = x + cubeSize;
	}

	std::cout << x << "," << y << "," << z << std::endl;
	int cellCount = 0;
	for (int i = 0;i < cubeX;i++) {
		for (int j = 0;j < cubeY;j++) {
			for (int k = 0;k < cubeZ;k++) {
				if (cube[i][j][k].value == 1)
					cellCount++;
			}

		}
	}
	std::cout << cellCount << std::endl;
	MarchingCubes mc(particlePosition,particleDensity,pData.mass,pData,particleCount);
	int currentCube[8];
	x = centerX -width/2;
	
	for (int i = 0;i < cubeX - 1;i++) {
		y = centerY - length/2;
		for (int j = 0;j < cubeY - 1;j++) 
		{
			z = centerZ - height / 2;
			for (int k = 0;k < cubeZ - 1;k++) 
			{
				currentCube[0] = cube[i][j][k].value;currentCube[1] = cube[i][j][k + 1].value;currentCube[2] = cube[i][j + 1][k].value;
				currentCube[3] = cube[i][j + 1][k + 1].value;currentCube[4] = cube[i + 1][j][k].value;currentCube[5] = cube[i + 1][j][k + 1].value;
				currentCube[6] = cube[i + 1][j + 1][k].value;currentCube[7] = cube[i + 1][j + 1][k + 1].value;
				mc.run(i,j,k,cube,currentCube,glm::vec3(x,y,z),cubeSize,150,cube[i][j][k].values);
				z += cubeSize;
			}
			y += cubeSize;
		}
		x += cubeSize;
	}

}



bool SphContainer::checkParticleExists(float x, float y, float z,float cubeSize,float &density,int i,int j,int k)
{
	float thresholdDistance = 0.01f;
	glm::vec3 pos(x, y, z);
	bool particleExists = false;
	for (int i = 0;i < particleCount;i++)
	{
		glm::vec3 particlePos(particlePosition[i].x, particlePosition[i].y, particlePosition[i].z);
		if (abs(particlePos.x - x) <= cubeSize && abs(particlePos.y - y) <= cubeSize && abs(particlePos.z - z) <= cubeSize) {
			
			return true;
		}
		/*float distance = glm::distance2(particlePos, pos);
		distance = sqrt(distance);
		if (distance < thresholdDistance)
			return true;*/

	}
	return false;
}

double SphContainer::getColorFieldValue(glm::vec3 point)
{
	float h2 = pData.r2;
	float h = pData.r;
	double cf = 0.0;
	int count = 0;
	for (int i = 0;i < (int)particleCount;i++)
	{
		float dist2 = glm::distance2(point, glm::vec3(particlePosition[i].x, particlePosition[i].y, particlePosition[i].z));
		if (sqrt(dist2) > h)
			continue;
		double tempVal = ((h2 - dist2)*(h2 - dist2)*(h2 - dist2))/particleDensity[i];
		cf += tempVal;
		
	}
	cf *= pData.mass;
	cf = (315.0 / (64.0*3.14*pow(h, 9)))*cf;
	return cf;
}

void SphContainer::calculateDensities(int i, int j, int k)
{
	float dist;
	glm::vec3 particle;
	for (int l = 0;l < particleCount;l++)
	{
		particle.x = particlePosition[l].x;
		particle.y = particlePosition[l].y;
		particle.z = particlePosition[l].z;
		dist = glm::distance(cube[i][j][k].pos[0], particle);
		if (dist < width / 32)
		{
			cube[i][j][k].values[0] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[1], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[1] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[2], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[2] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[3], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[3] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[4], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[4] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[5], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[5] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[6], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[6] += (particleDensity[i] / dist)*0.001;
		}
		dist = glm::distance(cube[i][j][k].pos[7], particle);
		if (dist < width / 30)
		{
			cube[i][j][k].values[7] += (particleDensity[i] / dist)*0.001;
		}
	}
}