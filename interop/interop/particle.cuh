#ifndef PARTICLE
#define PARTICLE

#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdio.h>

#include "glew\glew.h"
#include "freeglut\freeglut.h"

#include <cuda_gl_interop.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;

#define numMaterials 4
#define numParticle 300
//===================================================================
class Material{
public:
	float mass,
		restDensity,
		stiffness,
		bulkViscosity,
		surfaceTension,
		kElastic,
		maxDeformation,
		meltRate,
		viscosity,
		damping,
		friction,
		stickiness,
		smoothing,
		gravity;
	int materialIndex;

	__host__ __device__ Material();
};
//===================================================================
class Particle{
public:
	__host__ __device__ void update();
	__host__ __device__ void draw();
	__host__ __device__ Particle(Material* mat);
	__host__ __device__ Particle(Material* mat, float x, float y);
	__host__ __device__ Particle(Material* mat, float x, float y, float4 c);
	__host__ __device__ Particle(Material* mat, float x, float y, float u, float v);
	__host__ __device__  void initializeWeights(int gSizeY);

public:
	float3		pos;
	float3      trail;
	float4		color;

	Material* mat;
	float x, y, u, v, gu, gv, T00, T01, T11;
	int cx, cy, gi;
	float px[3];
	float py[3];
	float gx[3];
	float gy[3];
};
//===================================================================
class Node{
public:
	float mass,
		particleDensity,
		gx,
		gy,
		u,
		v,
		u2,
		v2,
		ax,
		ay;
	float cgx[numMaterials];
	float cgy[numMaterials];
	bool active;
	__host__ __device__ Node();
};
//===================================================================
class ParticleController{
public:
	__host__ __device__ float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
	__host__ __device__ ParticleController();
	__host__ __device__ void initializeGrid(int sizeX, int sizeY);
	__host__ __device__ void addParticles();
	__host__ __device__ void update();
	string print(int count);
	void getVectors();

	__device__ float cuda_uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
	__device__ void cuda_update();
	__device__ void cuda_getVectors();
	__device__ void device_initializeGrid(int sizeX, int sizeY);
	__device__ void device_addParticles();
	__device__ void ParticleController::PreparePrint(string text);

public:
	int gSizeX,
		gSizeY,
		gSizeY_3,
		scale;
	Node* grid;
	bool* active;
	Particle *particles;
	int nParticles;
	Material materials[3];
	Material* P_Materials;
	float2* vectors;
	string printBuffer;
};

#endif