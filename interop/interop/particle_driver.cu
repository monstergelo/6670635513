#include "particle.cuh"
using namespace std;

ParticleController* g_particles;
ParticleController particles;

class Ball{
public:
	int content = 33;
	int* collection;

	__host__ __device__
		void init(){
		content = 55;
		collection = (int*)malloc(sizeof(int) * 5000);

		for (int i = 0; i < 5000; i++){
			collection[i] = i;
		}
	};

	__host__ __device__
		Ball(){
		content = 99;
	};
};

class Box{
public:
	int* collection;
	int temp;
	Ball* ballInTheBox;
	int content = 3;

	__host__ __device__
		void init(){
		content = 5;
		ballInTheBox = (Ball*)malloc(sizeof(Ball)*5000);
		ballInTheBox[0] = Ball();
		ballInTheBox[0].init();
		content = ballInTheBox[0].content;

		for (int i = 0; i < 5000; i++){
			collection[i] = ballInTheBox[0].collection[i];
		}
		
		temp = collection[1543];
	}
};

class ParticleControllerWrapper{
public:
	ParticleController p;
	float3* pos;
};

__global__
void testBox(Box* b){
	b->init();
}

__global__
void initWrapper(ParticleControllerWrapper* wrap){
	wrap->p.initializeGrid(400, 200);
	wrap->p.addParticles();
	wrap->p.scale = 4.0f;

	for (int i = 0; i < 300; i++){
		float x = wrap->p.particles[i].pos.x;
		float y = wrap->p.particles[i].pos.y;
		float z = wrap->p.particles[i].x;
		wrap->pos[i] = make_float3(x, y, z);
	}
}

__global__
void wrapperUpdate(ParticleControllerWrapper* wrap){
	wrap->p.update();

	for (int i = 0; i < 300; i++){
		float x = wrap->p.particles[i].pos.x;
		float y = wrap->p.particles[i].pos.y;
		float z = wrap->p.particles[i].x;
		wrap->pos[i] = make_float3(x, y, z);
	}
}

int awmain(){
	ParticleController particles;
	ParticleControllerWrapper* d_particles;
	float3* d_particles_position;
	float3* tempC;
	tempC = (float3*)malloc(sizeof(float3) * 300);

	particles.initializeGrid(400, 200);
	particles.addParticles();
	particles.scale = 4.0f;

	//device stuff
	cudaMalloc((void**)&d_particles, sizeof(ParticleControllerWrapper));
	cudaMalloc((void**)&d_particles_position, sizeof(float3) * 300);
	cudaMemcpy(&(d_particles->pos), &d_particles_position, sizeof(float3*), cudaMemcpyHostToDevice);

	initWrapper << <1, 1 >> >(d_particles);
	cudaDeviceSynchronize();

	int i = 0;
	while (1){
		//device version
		cudaMemcpy(tempC, d_particles_position, sizeof(float3) * 300, cudaMemcpyDeviceToHost);
		cout << "Wewewewe" << endl;
		string a;

		a.append("[");
		for (int i = 0; i < 100; i++)
		{
			a.append("(");

			std::ostringstream streamObj3;
			streamObj3 << std::fixed;
			streamObj3 << std::setprecision(0);
			streamObj3 << tempC[i].x;
			streamObj3 << "|";
			streamObj3 << tempC[i].y;
			streamObj3 << "|";
			streamObj3 << tempC[i].z;
			std::string strObj3 = streamObj3.str();
			a.append(strObj3);

			a.append("),");
		}
		a.append("]");
		cout << a << endl;
		wrapperUpdate << <1, 1 >> >(d_particles);
		cudaDeviceSynchronize();

		//host version
		cout << "cycle " << i << "-----------------------------------------------" << endl;
		cout << particles.print(100);
		i++;
		char b = getchar();
		particles.update();
	}

	cout << "end";
	
	

	return 0;
}