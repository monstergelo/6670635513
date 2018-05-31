//#include "particle.cuh"
//
//
//// constants
//const unsigned int g_window_width = 512;
//const unsigned int g_window_height = 512;
//
//const unsigned int g_mesh_width = 256;
//const unsigned int g_mesh_height = 256;
//
//int vectorCount;
//GLuint vbo;
//struct cudaGraphicsResource *vbo_cuda;
//
////method declaration
//void display(void);
//void mouse(int button, int state, int x, int y);
//void motion(int x, int y);
//void keyboard(unsigned char key, int, int);
//
//__global__ void square_coordinate(float2 *vectors);
//__global__ void dummyCoordinate(float2* vectors);
//
//class Dummy{
//public:
//	float2* pos;
//	int size;
//	Dummy(float2* input, int input2){
//		pos = input;
//		size = input2;
//	}
//
//	__device__
//		void move(){
//		for (int i = 0; i < size; i++)
//		{
//			pos[i].x += 0.01;
//		}
//	}
//
//private:
//};
//
//Dummy *d;
//Dummy *d_g;
//ParticleController *cuda_particles;
//float2 *changes;
//
//__global__
//void justForTest(ParticleController* device, int count, float2* result){
//	for (int i = 0; i < count; i++){
//		result[i] = make_float2(device->particles[i].pos.x, device->particles[i].pos.y);
//	}
//}
//
////Actual program------------------------------------------------------------------------------------
//int wamain(int argc, char** argv)
//{
//	// Create GL context
//	glutInit(&argc, argv);
//
//	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//	glutInitWindowSize(g_window_width, g_window_height);
//	glutCreateWindow("interop");
//
//	GLenum glewInitResult = glewInit();
//	if (glewInitResult != GLEW_OK) {
//		throw std::runtime_error("Couldn't initialize GLEW");
//	}
//
//
//	// initialize GL
//	glClearColor(0.0, 0.0, 0.0, 1.0);
//	glDisable(GL_DEPTH_TEST);
//
//	// viewport
//	glViewport(0, 0, g_window_width, g_window_height);
//
//	// projection
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	glOrtho(0, g_window_width, g_window_height, 0, -10, 10);
//
//	cudaGLSetGLDevice(0);
//
//
//	// register callbacks
//	glutDisplayFunc(display);
//	glutKeyboardFunc(keyboard);
//
//	//initialize particles
//	ParticleController particles;
//	particles.initializeGrid(g_window_width, g_window_height);
//	particles.addParticles();
//	particles.scale = 1.0f;
//	int particle_count = 300;
//
//	cout << particles.particles[200].pos.x;
//	cout << particles.print(100);
//
//	for (int iii = 0; iii < 100; iii++){
//		particles.update();
//	}
//	
//
//	Node* g_grid;
//	bool* g_active;
//	Particle* g_particles;
//	Material* g_material;
//	Material* g_particles_material;
//
//	cudaMalloc((void**)&cuda_particles, sizeof(ParticleController));
//	cudaMemcpy(cuda_particles, &particles, sizeof(ParticleController), cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&g_particles, sizeof(Particle)*particle_count);
//	cudaMemcpy(g_particles, particles.particles, sizeof(Particle)*particle_count, cudaMemcpyHostToDevice);
//	cudaMemcpy(&(cuda_particles->particles), &(g_particles), sizeof(Particle*), cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&g_material, sizeof(Material)*particle_count);
//	cudaMemcpy(g_material, particles.P_Materials, sizeof(Material)*particle_count, cudaMemcpyHostToDevice);
//	cudaMemcpy(&(cuda_particles->materials), &(g_material), sizeof(Material*), cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&g_grid, sizeof(Node)*g_window_width*g_window_height);
//	cudaMemcpy(g_grid, particles.grid, sizeof(Node)*g_window_width*g_window_height, cudaMemcpyHostToDevice);
//	cudaMemcpy(&(cuda_particles->grid), &(g_grid), sizeof(Node*), cudaMemcpyHostToDevice);
//
//
//
//	// create vbo
//	vectorCount = particle_count;
//	unsigned int size = vectorCount * sizeof(float2);
//	glGenBuffers(1, &vbo);
//
//	// bind, initialize, unbind
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//	// register buffer object with CUDA
//	cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);
//
//	glutMainLoop();
//
//	cudaFree(cuda_particles);
//	cudaFree(g_active);
//	cudaFree(g_grid);
//	cudaFree(g_material);
//	cudaFree(g_particles);
//
//	return 0;
//}
//
//__global__
//void changeVectorParticles(float2* vectors, ParticleController* d_g, int count){
//	d_g->cuda_update();
//	Particle* changes = d_g->particles;
//	for (int i = 0; i < count; i++){
//		//vectors[i] = make_float2(changes[i].pos.x, changes[i].pos.y*(0.3));
//		vectors[i] = make_float2(50.0f*i, 50.0*i);
//	}
//}
//
//void display(void)
//{
//	float2 *raw_ptr;
//	size_t buf_size;
//
//	cudaGraphicsMapResources(1, &vbo_cuda, 0);
//	cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);
//
//	changeVectorParticles << <1, 1 >> >(raw_ptr, cuda_particles, vectorCount);
//
//	cudaGraphicsUnmapResources(1, &vbo_cuda, 0);
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	// render from the vbo
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	glVertexPointer(2, GL_FLOAT, 0, 0);
//
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glColor3f(1.0, 0.0, 0.0);
//	glDrawArrays(GL_POINTS, 0, vectorCount);
//	glDisableClientState(GL_VERTEX_ARRAY);
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//	glutSwapBuffers();
//	glutPostRedisplay();
//}
//
//
//void keyboard(unsigned char key, int, int)
//{
//	switch (key)
//	{
//	case(27) :
//		// deallocate memory
//		//g_vec.clear();
//		//g_vec.shrink_to_fit();
//		exit(0);
//	default:
//		break;
//	}
//}