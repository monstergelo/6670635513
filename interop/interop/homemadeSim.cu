/*
use ParticleData
use FFI, only: iomsg, iow1, iow2
use DataIn
use DataOut
implicit none
call InputPara() ! Input data
call calcEnergy() ! Calculate kinetic energy
*/
#include "particle.cuh"
#include <stdio.h>     
#include <math.h>
#include <stdlib.h>     
#include <time.h> 
#include <iostream> 
using namespace std;

class Grid{
public:
	float pos[3];	//position on x-y-z
	bool fixPos[3];	//wether position fixed on axis x-y-z
	float Mg;		//mass of node
	float PXg[3];	//momentum on axis x-y-z
	float FXg[3]; 	//force on axis x-y-z

	//contact??
	float ndir[3];
	float sdir[3];

	Grid(){
		for (int i = 0; i<3; i++){
			pos[i] = 0;
			fixPos[i] = false;
			PXg[i] = 0;
			FXg[i] = 0;
		}
		Mg = 0;
	}
};
//=================================================================================
class Ball{
public:
	float XX[3]; 	//position for drawing x-y-z
	float Xp[3]; 	//position for calculation x-y-z
	float VXp[3];	//particle velocity x-y-z
	float FXp[3];	//particle load/force x-y-z

	float vol;		//volume
	float sig_y;	//yield stress
	float SM;		//mean stress
	float Seqv;		//mises stress
	float SDxx, SDyy, SDzz, SDxy, SDyz, SDxz;//deviatoric stress 
	float epeff;	//effective plastic strain
	float celcius_t;//temperature

	bool Skip;		//flag to skip from process
	bool failure;	//flag for failure
	int icell;		//cell number,for grid?
	float dmg;		//damage
	float lt;		//lighting time
	float ie;		//internal energy
	float mass;		//mass
	float cp;		//sound speed
};

//variable
Ball particlelist[30];
int w = 400;
int h = 400;
Grid gridlist[1600];
bool debug = false;
float passedTime;
GLuint ballvbo;
struct cudaGraphicsResource *ball_vbo_cuda;
int ballCount = 30;
float2 *deviceBatch;
float2 *newBatch;

const unsigned int g_window_width = 512;
const unsigned int g_window_height = 512;

//=================================================================================
void createGrid(){
	int collumn = w / 10;
	int row = h / 10;

	for (int i = 0; i<collumn; i++){
		for (int j = 0; j<row; j++){
			gridlist[j + i*row].pos[0] = i * 10;
			gridlist[j + i*row].pos[1] = j * 10;

			if ((i == 0) || (i == collumn-1) || (j == 0) || (j == row-1)){
				gridlist[j + i*row].fixPos[0] = true;
				gridlist[j + i*row].fixPos[1] = true;
				gridlist[j + i*row].fixPos[2] = true;
			}
		}
	}
}
//=================================================================================
void createParticle(){
	srand(time(NULL));

	for (int i = 0; i<30; i++){
		particlelist[i].XX[0] = rand() % w;
		particlelist[i].XX[1] = rand() % h;

		particlelist[i].Xp[0] = particlelist[i].XX[0];
		particlelist[i].Xp[1] = particlelist[i].XX[1];

		particlelist[i].mass = 1;
		particlelist[i].VXp[0] = 10;
		particlelist[i].VXp[1] = 0;
	}
}
//=================================================================================
void MapParticleToGrid(){
	for (int i = 0; i<30; i++){
		int shortest = 999999;
		int shortest_index = 0;

		int x1 = particlelist[i].Xp[0];
		int y1 = particlelist[i].Xp[1];

		for (int ii = 0; ii<1200; ii++){
			int x2 = gridlist[ii].pos[0];
			int y2 = gridlist[ii].pos[1];

			int delta_X = x2 - x1;
			int delta_Y = y2 - y1;
			int distance = sqrt((delta_X*delta_X) + (delta_Y*delta_Y));

			if (distance < shortest){
				shortest = distance;
				shortest_index = ii;
			}
		}

		particlelist[i].icell = shortest_index;
	}
}
//=================================================================================
void CheckGridMomentumInitial(){
	if (debug){
		cout << "====Particle=====" << endl;
		for (int i = 0; i<30; i++){
			if (particlelist[i].icell == 0)
				cout << "wawa ";

			cout << particlelist[i].Xp[0] << "," << particlelist[i].Xp[1] << endl;
		}

		cout << "====Grid=====" << endl;
		for (int i = 0; i<1200; i++){
			if (gridlist[i].Mg != 0)
				cout << gridlist[i].pos[0] << "," << gridlist[i].pos[1] << ":" << gridlist[i].Mg << endl;
		}
	}
}
//=================================================================================
void GridMomentumInitial(){
	MapParticleToGrid();

	for (int i = 0; i<30; i++){
		int gridIndex = particlelist[i].icell;
		gridlist[gridIndex].Mg += particlelist[i].mass;

		gridlist[gridIndex].PXg[0] += particlelist[i].mass*particlelist[i].VXp[0];
		gridlist[gridIndex].PXg[1] += particlelist[i].mass*particlelist[i].VXp[1];
		gridlist[gridIndex].PXg[2] += particlelist[i].mass*particlelist[i].VXp[2];
	}

	CheckGridMomentumInitial();
}
//=================================================================================
void ApplyBoundaryConditions(){
	for (int i = 0; i<1200; i++){
		//static on axis-x
		if (gridlist[i].fixPos[0]){
			gridlist[i].PXg[0] = 0;
			gridlist[i].FXg[0] = 0;
		}

		//static on axis-y
		if (gridlist[i].fixPos[1]){
			gridlist[i].PXg[1] = 0;
			gridlist[i].FXg[1] = 0;
		}

		//static on axis-z
		if (gridlist[i].fixPos[2]){
			gridlist[i].PXg[2] = 0;
			gridlist[i].FXg[2] = 0;
		}
	}
}
//=================================================================================
void CheckGridMomentumUpdate(){
	if (debug){
		for (int i = 0; i<1200; i++){
			if (gridlist[i].FXg[1] != 0)
				cout << gridlist[i].pos[0] << "," << gridlist[i].pos[1] << ":" << gridlist[i].FXg[1] << endl;
		}
	}
}

void GridMomentumUpdate(){
	float sxx;
	float syy;
	float szz;
	float sxy;
	float syz;
	float sxz;

	float fx[3]; //external force
	float fi[3]; //internal force
	float gravity = 9.8;

	for (int i = 0; i<30; i++){
		sxx = particlelist[i].SM + particlelist[i].SDxx;
		syy = particlelist[i].SM + particlelist[i].SDyy;
		szz = particlelist[i].SM + particlelist[i].SDzz;
		sxy = particlelist[i].SDxy;
		syz = particlelist[i].SDyz;
		sxz = particlelist[i].SDxz;

		//external force
		fx[0] = particlelist[i].FXp[0];
		fx[1] = particlelist[i].FXp[1] + particlelist[i].mass*gravity;
		fx[2] = particlelist[i].FXp[2];

		fi[0] = -(sxx + sxy + sxz)*particlelist[i].vol;
		fi[1] = -(sxy + syy + syz)*particlelist[i].vol;
		fi[2] = -(sxz + syz + szz)*particlelist[i].vol;

		gridlist[particlelist[i].icell].FXg[0] += fi[0] + fx[0];
		gridlist[particlelist[i].icell].FXg[1] += fi[1] + fx[1];
		gridlist[particlelist[i].icell].FXg[2] += fi[2] + fx[2];
	}

	CheckGridMomentumUpdate();
}
//=================================================================================
void CheckIntegrateMomentum(){
	if (debug){
		for (int i = 0; i<1200; i++){
			if (gridlist[i].PXg[1] != 0)
				cout << gridlist[i].pos[0] << "," << gridlist[i].pos[1] << ":" << gridlist[i].PXg[1] << endl;
		}
	}
}

void IntegrateMomentum(){
	for (int i = 0; i<1200; i++){
		gridlist[i].PXg[0] += gridlist[i].FXg[0] * passedTime;
		gridlist[i].PXg[1] += gridlist[i].FXg[1] * passedTime;
		gridlist[i].PXg[2] += gridlist[i].FXg[2] * passedTime;
	}

	ApplyBoundaryConditions();
	CheckIntegrateMomentum();
}
//=================================================================================
void Lagr_NodContact(){
	for(int i=0; i<1200; i++){
		for(int ii=i+1; ii<1200; ii++){
			float nx;
			float ny;
			float nz;

			nx = gridlist[i].ndir[0] - gridlist[ii].ndir[0] ;
			ny = gridlist[i].ndir[1] - gridlist[ii].ndir[0] ;
			nz = gridlist[i].ndir[2] - gridlist[ii].ndir[0] ;

			float distance = sqrt(nx*nx + ny*ny + nz*nz);
			nx = nx/distance;
			ny = ny/distance;
			nz = nz/distance;

			gridlist[i].ndir[0] = nx;
			gridlist[i].ndir[1] = ny;
			gridlist[i].ndir[2] = nz;
			gridlist[ii].ndir[0] = nx*(-1);
			gridlist[ii].ndir[1] = ny*(-1);
			gridlist[ii].ndir[2] = nz*(-1);

			float x_component = gridlist[i].PXg[0]*gridlist[ii].Mg - gridlist[ii].PXg[0]*gridlist[i].Mg; 
			float y_component = gridlist[i].PXg[1]*gridlist[ii].Mg - gridlist[ii].PXg[1]*gridlist[i].Mg; 
			float z_component = gridlist[i].PXg[2]*gridlist[ii].Mg - gridlist[ii].PXg[2]*gridlist[i].Mg; 
			float crit = nx*(x_component) + ny*(y_component) + nz*(z_component);

			float nomforce = crit/distance;
			float contactForce[3];
			contactForce[0] = nomforce*gridlist[i].ndir[0];
			contactForce[1] = nomforce*gridlist[i].ndir[1];
			contactForce[2] = nomforce*gridlist[i].ndir[2];

			gridlist[i].FXg[0] -= contactForce[0];
			gridlist[i].FXg[1] -= contactForce[1];
			gridlist[i].FXg[2] -= contactForce[2];

			gridlist[ii].FXg[0] += contactForce[0];
			gridlist[ii].FXg[1] += contactForce[1];
			gridlist[ii].FXg[2] += contactForce[2];

			gridlist[i].PXg[0] -= contactForce[0] * passedTime;
			gridlist[i].PXg[1] -= contactForce[1] * passedTime;
			gridlist[i].PXg[2] -= contactForce[2] * passedTime;

			gridlist[ii].PXg[0] += contactForce[0] * passedTime;
			gridlist[ii].PXg[1] += contactForce[1] * passedTime;
			gridlist[ii].PXg[2] += contactForce[2] * passedTime;
		}
	}
}
//=================================================================================
void CheckParticlePositionUpdate(){
	if (debug){
		for (int i = 0; i<30; i++){
			cout << particlelist[i].XX[0] << "," << particlelist[i].XX[1] << endl;
		}

		cout << "+================================+" << endl;
	}

}

void ParticlePositionUpdate(){
	CheckParticlePositionUpdate();
	for (int i = 0; i<30; i++){
		float v[3];
		float a[3];
		float mass = gridlist[particlelist[i].icell].Mg;

		v[0] = gridlist[particlelist[i].icell].PXg[0] / mass;
		v[1] = gridlist[particlelist[i].icell].PXg[1] / mass;
		v[2] = gridlist[particlelist[i].icell].PXg[2] / mass;

		a[0] = gridlist[particlelist[i].icell].FXg[0] / mass;
		a[1] = gridlist[particlelist[i].icell].FXg[1] / mass;
		a[2] = gridlist[particlelist[i].icell].FXg[2] / mass;

		particlelist[i].XX[0] += v[0] * passedTime;
		particlelist[i].XX[1] += v[1] * passedTime;
		particlelist[i].XX[2] += v[2] * passedTime;

		particlelist[i].VXp[0] += a[0] * passedTime;
		particlelist[i].VXp[1] += a[1] * passedTime;
		particlelist[i].VXp[2] += a[2] * passedTime;

		particlelist[i].Xp[0] = particlelist[i].XX[0];
		particlelist[i].Xp[1] = particlelist[i].XX[1];
		particlelist[i].Xp[2] = particlelist[i].XX[2];
	}
	CheckParticlePositionUpdate();
}
//=================================================================================
void CheckGridMomentumMUSL(){
	if (debug){
		for (int i = 0; i<1200; i++){
			if (gridlist[i].PXg[1] != 0)
				cout << gridlist[i].pos[0] << "," << gridlist[i].pos[1] << ":" << gridlist[i].PXg[1] << endl;
		}
	}
}

void GridMomentumMUSL(){
	for (int i = 0; i<30; i++){
		gridlist[particlelist[i].icell].PXg[0] += particlelist[i].VXp[0];
		gridlist[particlelist[i].icell].PXg[1] += particlelist[i].VXp[1];
		gridlist[particlelist[i].icell].PXg[2] += particlelist[i].VXp[2];
	}

	CheckGridMomentumMUSL();
}
//=================================================================================

__global__
void changeVectorParticles(float2* vectors, int count, float2* news){
	for (int i = 0; i < count; i++){
		float x = news[i].x;
		float y = news[i].y;
		vectors[i] = make_float2(x, y);
	}
}
void updateBall(){
	//! Step 1: Initialize background grid nodal mass and Momentum
	GridMomentumInitial();

	//! Step 2: Apply boundary conditions
	ApplyBoundaryConditions();

	//! Step 3: Update particles stress (Only For USF)
	//ParticleStressUpdate()

	//! Step 4: Calculate the grid nodal force
	GridMomentumUpdate();

	//! Step 5: Integrate momentum equations on background grids
	IntegrateMomentum();

	//! Step 6: Detect contact grid node, calculate contact force
	//Lagr_NodContact()

	//! Step 7: Update particles position and velocity
	ParticlePositionUpdate();

	//! Step 8: Recalculate the grid node momentum for MUSL
	GridMomentumMUSL();
	ApplyBoundaryConditions();

	//! Step 9: Update particles stress for both USF and MUSL
	//ParticleStressUpdate()
}
void makeBatch(float2* result){
	result = (float2*)malloc(sizeof(float2));
	for (int i = 0; i < ballCount; i++){
		float x = particlelist[i].XX[0];
		float y = particlelist[i].XX[1];
		result[i] = make_float2(x, y);
	}
}
void display(void)
{
	float2 *raw_ptr;
	size_t buf_size;

	updateBall();
	for (int i = 0; i < ballCount; i++){
		float x = particlelist[i].XX[0];
		float y = particlelist[i].XX[1];
		newBatch[i] = make_float2(x, y);
	}
	cudaMemcpy(deviceBatch, newBatch, sizeof(float2)*ballCount, cudaMemcpyHostToDevice);

	cudaGraphicsMapResources(1, &ball_vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, ball_vbo_cuda);

	changeVectorParticles << <1, 1 >> >(raw_ptr, ballCount, deviceBatch);

	cudaGraphicsUnmapResources(1, &ball_vbo_cuda, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, ballvbo);
	glVertexPointer(2, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, ballCount);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();
	glutPostRedisplay();
}
int main(int argc, char** argv){
	createGrid();
	createParticle();
	passedTime = 0.01;

	cudaMalloc((void**)&deviceBatch, sizeof(float2)*ballCount);
	newBatch = (float2*)malloc(sizeof(float2)*ballCount);


	// Create GL context
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(g_window_width, g_window_height);
	glutCreateWindow("interop");

	GLenum glewInitResult = glewInit();
	if (glewInitResult != GLEW_OK) {
		throw std::runtime_error("Couldn't initialize GLEW");
	}


	// initialize GL
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, g_window_width, g_window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, g_window_width, g_window_height, 0, -10, 10);

	cudaGLSetGLDevice(0);

	// register callbacks
	glutDisplayFunc(display);

	// create vbo
	unsigned int size = ballCount * sizeof(float2);
	glGenBuffers(1, &ballvbo);

	// bind, initialize, unbind
	glBindBuffer(GL_ARRAY_BUFFER, ballvbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(&ball_vbo_cuda, ballvbo, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

	//bool end = false;
	//while (!end){
	
	//}
}

/*
Possible error:
1. Grid parameter belom direset
2. contact
*/