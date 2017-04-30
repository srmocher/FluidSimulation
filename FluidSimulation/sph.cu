#include "sph.cuh"
#include <helper_math.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#define DIRECTION_COUNT 27


__constant__ float PI = 3.14159265359;
__constant__ float airFriction = 1.0;

__constant__ int dx[] = {0, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
__constant__ int dy[] = {0, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0, -1, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1 };
__constant__ int dz[] = {0,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1 };



texture<float, 1, cudaReadModeElementType> texture_density;
texture<float4, 1, cudaReadModeElementType> texPos;
texture<int, 1, cudaReadModeElementType> texStartHsh;
texture<int, 1, cudaReadModeElementType> texEndHsh;

__constant__ particleData PARTICLE_DATA[1];
__device__ void checkBoundary(int cId);

int blockCount;
int* blockGPUCount;



__device__ float getNorm(float3 obj)
{
    return sqrt(obj.x * obj.x + obj.y * obj.y + obj.z * obj.z);
}
__device__ float getSqNorm(float3 obj)
{
    return obj.x * obj.x + obj.y * obj.y + obj.z * obj.z;
}

void bindToTextures(particleData* pData)
{
    cudaBindTexture(NULL, texture_density, pData->dens, pData->count * sizeof(float));    
    cudaBindTexture(NULL, texPos, pData->posTextured, pData->count * sizeof(float) * 4);
    cudaBindTexture(NULL, texStartHsh, pData->hashTableStart, pData->HASH_TABLE_SIZE * sizeof(int));
    cudaBindTexture(NULL, texEndHsh, pData->hashTableEnd, pData->HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&blockGPUCount, sizeof(int));
}

__device__ int3 getBlock(float3 pos)
{
    int3 rv;
    rv.x = pos.x / PARTICLE_DATA->gridDimConst;
    rv.y = pos.y / PARTICLE_DATA->gridDimConst;
    rv.z = pos.z / PARTICLE_DATA->gridDimConst;
    return rv;
}

__device__ int getBlockHash(int3 bl)
{
    uint hsh = (((bl.x * 73856093) + (bl.y * 19349663) + (bl.z * 83492791))) % PARTICLE_DATA->HASH_TABLE_SIZE;
    return hsh;        
}

__device__ unsigned int getParticleHash(float3 bl)
{
    float d = PARTICLE_DATA->gridDimConst;
    uint hsh = ((( (int) (bl.x / d) * 73856093) + ( (int) (bl.y / d) * 19349663) + ( (int) (bl.z / d) * 83492791)))
        % PARTICLE_DATA->HASH_TABLE_SIZE;         
    return hsh;
}

__global__ void computeDensities()
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= PARTICLE_DATA->count)
        return;

        unsigned int blhsh;
        int3 block, tempBlock;
        int* hashTableStart = PARTICLE_DATA->hashTableStart;
        int* hashTableEnd = PARTICLE_DATA->hashTableEnd;
        int* pInd = PARTICLE_DATA->pind;

        float3 vc, ms;
        float3* pos = PARTICLE_DATA->pos;
        float4 ps4 = tex1Dfetch(texPos, id);
        float3 ps = {ps4.x, ps4.y, ps4.z};
        float rd;
        float r2 = PARTICLE_DATA->r2;
        block = getBlock(ps);
        float densLoc = r2 * r2 * r2;
        float dtr;
        
        int totalCount = 0;
        
        for(int dir = 0; dir < DIRECTION_COUNT; ++dir)
        {
            
            tempBlock.x = block.x + dx[dir];
            tempBlock.y = block.y + dy[dir];
            tempBlock.z = block.z + dz[dir];
            blhsh = getBlockHash(tempBlock);  
            int strt = tex1Dfetch(texStartHsh, blhsh);
            int end = tex1Dfetch(texEndHsh, blhsh);            
            if (strt >= PARTICLE_DATA->count)
                continue;          
            for (int j = strt; j <= end ; ++j)
            {   
              //  printf("NORMAL_DENSITY: %d dir: %d count: %d \n", id, dir, end - strt + 1);
                if (j == id)
                    continue;
                ps4 = tex1Dfetch(texPos, j);
                ms.x = ps4.x; ms.y = ps4.y; ms.z = ps4.z;
                vc = ps - ms;
                dtr = vc.x * vc.x + vc.y * vc.y + vc.z * vc.z;
                if (dtr > r2)
                    continue;
                ++totalCount;
                rd = r2 - dtr;
                densLoc += rd * rd * rd;              
            }   
            
        }   
        PARTICLE_DATA->dens[id] = PARTICLE_DATA->diffKern * (densLoc * PARTICLE_DATA->mass);    
}

__global__ void sph(float dt)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;
    
    float3* pos  = PARTICLE_DATA->pos;
    
    float3* vel  = PARTICLE_DATA->vel;
    float3* hVel = PARTICLE_DATA->hVel;
    float* dens  = PARTICLE_DATA->dens;
    int* pInd = PARTICLE_DATA->pind;
    int* hashTableStart = PARTICLE_DATA->hashTableStart;
    int* hashTableEnd = PARTICLE_DATA->hashTableEnd;
    float stiffness = PARTICLE_DATA->pressure_Koeff;
    
	
    float density = PARTICLE_DATA->rest_density;
	if (cId <= PARTICLE_DATA->count / 2) {
		density = 1000.0f;
	}
	else {
		density = 1000.0f;
	}

    float3 acceleration;
    float3 pressureForce;
    float3 viscosityForce;
    pressureForce.x = pressureForce.y = pressureForce.z = 0;
    viscosityForce.x = viscosityForce.y = viscosityForce.z = 0;
    acceleration.x = acceleration.y = acceleration.z = 0;

    float crDens = tex1Dfetch(texture_density, cId);
    float cpress = stiffness * (crDens - density);
    float express;
    float3 vc, ms;
    float4 ps4 = tex1Dfetch(texPos, cId);
    float3 ps = {ps4.x, ps4.y, ps4.z};
    float densTo;
    
    
    float3 vlc = vel[cId];
    float dst;
    float mult;
    
    int3 block, tempBlock;
    unsigned int blhsh;

    float r = PARTICLE_DATA->r;
    float r2 = PARTICLE_DATA->r2;
    block = getBlock(ps);    
    int totalCount = 0;
    for(int dir = 0; dir < DIRECTION_COUNT; ++dir)
    {
        tempBlock.x = block.x + dx[dir];
        tempBlock.y = block.y + dy[dir];
        tempBlock.z = block.z + dz[dir];
        blhsh = getBlockHash(tempBlock);
        int strt = tex1Dfetch(texStartHsh, blhsh);
        int end = tex1Dfetch(texEndHsh, blhsh);
        if (strt >= PARTICLE_DATA->count)
            continue;

        for (int j = strt; j <= end; ++j)
        {              
            if (j == cId)
                continue;
            ps4 = tex1Dfetch(texPos, j);
            densTo = tex1Dfetch(texture_density, j);

            ms.x = ps4.x; ms.y = ps4.y; ms.z = ps4.z;
            express = stiffness * (densTo - density);
            vc = ms - ps;
            dst = vc.x * vc.x + vc.y * vc.y + vc.z * vc.z;
            if (dst > r2)
                continue;
            ++totalCount;
            dst = sqrt(dst);
            mult = r - dst;
            pressureForce -=  vc * ((express + cpress) / (2 * dst) * mult * mult * mult);                  
            mult = mult / densTo;
            viscosityForce += (vel[j] - vlc) * mult*PARTICLE_DATA->visc[j];            
        } 
    }   


    pressureForce = pressureForce * (PARTICLE_DATA->pressKern * PARTICLE_DATA->mass);
    viscosityForce = viscosityForce * (PARTICLE_DATA->mass * PARTICLE_DATA->viscKern);    
    acceleration = (pressureForce + viscosityForce) * (1 / crDens);
    float clr = getNorm(acceleration) / 300;
    
	if (cId <= PARTICLE_DATA->count / 2) {
		PARTICLE_DATA->color[cId].x = clr;
		PARTICLE_DATA->color[cId].y = clr;
		PARTICLE_DATA->color[cId].z = 0.5 + clr;
	}
	else {
		PARTICLE_DATA->color[cId].x = clr + 0.5;
		PARTICLE_DATA->color[cId].y = clr;
		PARTICLE_DATA->color[cId].z = clr;
	}
    
    acceleration += PARTICLE_DATA->gravity;
    
    hVel[cId] += acceleration * dt;  
    PARTICLE_DATA->accel[cId] = acceleration;

}

__global__  void computeZindexes()
{
    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;
    uint pHash = getParticleHash(PARTICLE_DATA->pos[cId]);
    PARTICLE_DATA->zind[cId] = pHash;
    PARTICLE_DATA->pind[cId] = cId;    
}

__global__  void buildHashTable()
{    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;
    int pHash = PARTICLE_DATA->zind[cId];
    if (cId == 0)
    {
        PARTICLE_DATA->hashTableStart[pHash] = cId;
    }
    if (cId == PARTICLE_DATA->count - 1)
    {
        PARTICLE_DATA->hashTableEnd[pHash]= cId;
    }

    if (cId > 0 && pHash != PARTICLE_DATA->zind[cId - 1])
        PARTICLE_DATA->hashTableStart[pHash] = cId;
    if (cId < PARTICLE_DATA->count - 1 && pHash != PARTICLE_DATA->zind[cId + 1])
        PARTICLE_DATA->hashTableEnd[pHash] = cId;
}

__global__ void integratePositions(float dt)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;
    float4 posOld = tex1Dfetch(texPos, cId);
    float3 pos3 = {posOld.x, posOld.y, posOld.z};
    float3 vl = PARTICLE_DATA->hVel[cId];
    float3 fr = PARTICLE_DATA->accel[cId];
    double hdt = dt * 0.5;
    PARTICLE_DATA->vel[cId].x  = vl.x + fr.x * hdt;
    PARTICLE_DATA->vel[cId].y  = vl.y + fr.y * hdt;
    PARTICLE_DATA->vel[cId].z  = vl.z + fr.z * hdt;    
    PARTICLE_DATA->pos[cId] = pos3 + PARTICLE_DATA->hVel[cId] * dt;    
    checkBoundary(cId);
}

__device__ void checkBoundary(int cId)
{
    float3* pos = PARTICLE_DATA->pos + cId;
    float3 center = PARTICLE_DATA->center;
    float3 sizeContainer = PARTICLE_DATA->sizeContainer * 0.5;
    float3* hVel = PARTICLE_DATA->hVel + cId;
    float3* vel = PARTICLE_DATA->vel + cId;
    float wallDamping = PARTICLE_DATA->wallDamping;
    
    if (pos->x > sizeContainer.x + center.x)
    {
        pos->x = sizeContainer.x;
        hVel->x *= -wallDamping;
        vel->x *= -wallDamping;
    }
    if (pos->x < -sizeContainer.x + center.x)
    {
        pos->x = -sizeContainer.x + center.x;
        hVel->x *= -wallDamping;
        vel->x *= -wallDamping;
    }

      if (pos->y > sizeContainer.y + center.y)
    {
        pos->y = sizeContainer.y;
        hVel->y *= -wallDamping;
        vel->y *= -wallDamping;
    }
    if (pos->y < -sizeContainer.y + center.y)
    {
        pos->y = -sizeContainer.y + center.y;
        hVel->y *= -wallDamping;
        vel->y *= -wallDamping;
    }

    if (pos->z > sizeContainer.z + center.z)
    {
        pos->z = sizeContainer.z;
        hVel->z *= -wallDamping;
        vel->z *= -wallDamping;
    }
    if (pos->z < -sizeContainer.z + center.z)
    {
        pos->z = -sizeContainer.z + center.z;
        hVel->z *= -wallDamping;
        vel->z *= -wallDamping;
    }
}

__global__ void updateParticles()
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;

    int ind =  PARTICLE_DATA->pind[cId];
    float3 ps = PARTICLE_DATA->pos[ind];
    PARTICLE_DATA->posTextured[cId].x = ps.x; PARTICLE_DATA->posTextured[cId].y = ps.y; PARTICLE_DATA->posTextured[cId].z = ps.z; PARTICLE_DATA->posTextured[cId].w = 1;
    PARTICLE_DATA->tempVel[cId] = PARTICLE_DATA->vel[ind];
    PARTICLE_DATA->temphVel[cId] = PARTICLE_DATA->hVel[ind];
    PARTICLE_DATA->pind[cId] = cId;
}

__global__ void makeAlignedArray(int* blockCount, int perBlock)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->HASH_TABLE_SIZE)
        return;
    if (PARTICLE_DATA->hashTableStart[cId] > PARTICLE_DATA->count)
        return;
    
    int count = PARTICLE_DATA->hashTableEnd[cId] - PARTICLE_DATA->hashTableStart[cId] + 1;
    int start = PARTICLE_DATA->hashTableStart[cId];
    int newBlocks = (count + perBlock - 1) / perBlock;

    int pos = atomicAdd(blockCount, newBlocks);
    
    for (int i = 0; i < newBlocks; ++i)
    {
        PARTICLE_DATA->zind[pos + i] = start;
        PARTICLE_DATA->pind[pos + i] = min(perBlock, count);
        start += min(perBlock, count);
        count -= perBlock;
    }
}

__global__ void calculateForces(forceData frc, float dt)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->count)
        return;


    float4 ps = tex1Dfetch(texPos, cId);
    float3 hVel = PARTICLE_DATA->hVel[cId];
    float3 vel = PARTICLE_DATA->vel[cId];
    float3 vc = {ps.x - frc.coord.x, ps.y - frc.coord.y, ps.z - frc.coord.z};
    float dtt = dot(vc, vc);
    dt = dt * 3;
    if ( dtt < frc.r2)
    {
        float norm = sqrt(dtt);
        float d = frc.radius - norm;
        vc = vc * (1 / norm);
        PARTICLE_DATA->hVel[cId] = vc * frc.power;
        PARTICLE_DATA->vel[cId] = vc * frc.power;
    }
}
__global__ void initArrays()
{    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= PARTICLE_DATA->HASH_TABLE_SIZE)
        return;
    PARTICLE_DATA->hashTableStart[cId] = 1000 * 1000 * 1000;
}


void mySwap(float3* &a, float3* &b)
{
    float3* c = b;
    b = a;
    a = c;
}

void prepareFluidGPU(particleData& pData, float dt)
{
     computeZindexes<<<(pData.count + 255) / 256, 256>>>();  
     thrust::sort_by_key(thrust::device_ptr<int>(pData.zind), thrust::device_ptr<int>(pData.zind) + pData.count, thrust::device_ptr<int>(pData.pind));
     initArrays<<<(pData.HASH_TABLE_SIZE + 255) / 256, 256>>>();
     updateParticles<<<(pData.count + 255) / 256, 256>>>();
     buildHashTable<<<(pData.count + 255) / 256, 256>>>();
     
     mySwap(pData.vel, pData.tempVel);
     mySwap(pData.hVel, pData.temphVel);

}

void sph(particleData pData, float dt, forceData frc)
{
    int threads = 256;
    computeDensities<<<(pData.count + threads - 1) / threads, threads>>>();
    
    sph<<<(pData.count + threads - 1) / threads, threads>>>(dt);
    calculateForces<<<(pData.count + threads - 1) / threads, threads>>>(frc, dt);
    integratePositions<<<(pData.count + threads - 1) / threads, threads>>>(dt);    
}



void updateSimData(particleData& data)
{
    gpuErrchk( cudaMemcpyToSymbol(PARTICLE_DATA, &data, sizeof(data)));
}