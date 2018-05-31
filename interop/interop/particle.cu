#include "particle.cuh"

//===================================================================
Material::Material(){

	mass = 1;
	restDensity = 2;
	stiffness = 1;
	bulkViscosity = 1;
	surfaceTension = 0;
	kElastic = 0;
	maxDeformation = 0;
	meltRate = 0;
	viscosity = .02;
	damping = .001;
	friction = 0;
	stickiness = 0;
	smoothing = .02;
	gravity = .03;
}
//===================================================================
__host__ __device__
Node::Node() : mass(0), particleDensity(0), gx(0), gy(0), u(0), v(0), u2(0), v2(0), ax(0), ay(0), active(false) {
	memset(cgx, 0, 2 * numMaterials * sizeof(float));
}
//===================================================================
__host__ __device__
Particle::Particle(Material* mat) : pos(make_float3(0, 0, 0)), color(make_float4(.1, .5, 1, 1)), mat(mat), x(0), y(0), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
	memset(px, 0, 12 * sizeof(float));
}

__host__ __device__
Particle::Particle(Material* mat, float x, float y) : pos(make_float3(x, y, 0)), color(make_float4(.1, .5, 1, 1)), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
	memset(px, 0, 12 * sizeof(float));
}

__host__ __device__
Particle::Particle(Material* mat, float x, float y, float4 c) : pos(make_float3(x, y, 0)), color(c), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
	memset(px, 0, 12 * sizeof(float));
}

__host__ __device__
Particle::Particle(Material* mat, float x, float y, float u, float v) : pos(make_float3(x, y, 0)), color(make_float4(.1, .5, 1, 1)), mat(mat), x(x), y(y), u(u), v(v), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
	memset(px, 0, 12 * sizeof(float));
}

__host__ __device__
void Particle::initializeWeights(int gSizeY) {
	cx = (int)(x - .5f);
	cy = (int)(y - .5f);
	gi = cx * gSizeY + cy;

	float cx_x = cx - x;
	float cy_y = cy - y;

	// Quadratic interpolation kernel weights - Not meant to be changed
	px[0] = .5f * cx_x * cx_x + 1.5f * cx_x + 1.125f;
	gx[0] = cx_x + 1.5f;
	cx_x++;
	px[1] = -cx_x * cx_x + .75f;
	gx[1] = -2 * cx_x;
	cx_x++;
	px[2] = .5f * cx_x * cx_x - 1.5f * cx_x + 1.125f;
	gx[2] = cx_x - 1.5f;

	py[0] = .5f * cy_y * cy_y + 1.5f * cy_y + 1.125f;
	gy[0] = cy_y + 1.5f;
	cy_y++;
	py[1] = -cy_y * cy_y + .75f;
	gy[1] = -2 * cy_y;
	cy_y++;
	py[2] = .5f * cy_y * cy_y - 1.5f * cy_y + 1.125f;
	gy[2] = cy_y - 1.5f;
}

void CPUBurden(){
	int burden = 5000000;
	int arb = 1;
	for (int i = 0; i< burden; i++){
		arb = i ^ 2;
	}
}

//===================================================================
float ParticleController::uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v)
{
	float dx = x00 - x01;
	float dy = y00 - y10;
	float a = p01 - p00;
	float b = p11 - p10 - a;
	float c = p10 - p00;
	float d = y11 - y01;
	return ((((d - 2 * b - dy) * u - 2 * a + y00 + y01) * v +
		((3 * b + 2 * dy - d) * u + 3 * a - 2 * y00 - y01)) * v +
		((((2 * c - x00 - x10) * u + (3 * b + 2 * dx + x10 - x11)) * u - b - dy - dx) * u + y00)) * v +
		(((x11 - 2 * (p11 - p01 + c) + x10 + x00 + x01) * u +
		(3 * c - 2 * x00 - x10)) * u +
		x00) * u + p00;
}

ParticleController::ParticleController() : scale(1.0f) {
	//default
	materials[0].materialIndex = 0;
	materials[0].mass = 1.0f;
	materials[0].viscosity = 0.04f;

	materials[1].materialIndex = 1;
	materials[1].mass = 1.0f;
	materials[1].restDensity = 10.0f;
	materials[1].viscosity = 1.0f;
	materials[1].bulkViscosity = 3.0f;
	materials[1].stiffness = 1.0f;
	materials[1].meltRate = 1.0f;
	materials[1].kElastic = 1.0f;



	materials[2].materialIndex = 2;
	materials[2].mass = 0.7f;
	materials[2].viscosity = 0.03f;


	materials[3].materialIndex = 3;
}
void ParticleController::initializeGrid(int sizeX, int sizeY) {
	gSizeX = sizeX;
	gSizeY = sizeY;
	gSizeY_3 = sizeY - 3;
	grid = new Node[gSizeX*gSizeY];
	for (int i = 0; i < gSizeX*gSizeY; i++) {
		grid[i] = Node();
	}
}
void ParticleController::addParticles() {
	nParticles = 300;
	int p_index = 0;

	particles = (Particle*)malloc(sizeof(Particle)*nParticles);
	// Material 1
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			Particle p(&materials[0], i*.4 + 6, j*.8 / 5 + 6, make_float4(1, 0.5, 0.5, 1));
			p.initializeWeights(gSizeY);
			particles[p_index] = p;
			p_index++;
		}
	}

	// Material 2
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 5; j++) {
			Particle p(&materials[1], i*.4 + 150, j*.8 / 5 + 15, make_float4(1, 1, 1, 1));
			p.initializeWeights(gSizeY);
			particles[p_index] = p;
			p_index++;
		}
	}

	// Material 2
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			Particle p(&materials[2], i*.4 + 350, j*.8 / 5 + 15, make_float4(0.5, 1.0, 0.0, 1));
			p.initializeWeights(gSizeY);
			particles[p_index] = p;
			p_index++;
		}
	}

	P_Materials = (Material*)malloc(nParticles*sizeof(Material));
	for (int i = 0; i < nParticles; i++){
		P_Materials[i] = *(particles[i].mat);
	}
}

string ParticleController::print(int count){
	string a;

	a.append("[");
	for (int i = 0; i < count; i++)
	{
		Particle p = particles[i];
		a.append("(");

		std::ostringstream streamObj3;
		streamObj3 << std::fixed;
		streamObj3 << std::setprecision(0);
		streamObj3 << p.pos.x;
		streamObj3 << "|";
		streamObj3 << p.pos.y;
		streamObj3 << "|";
		streamObj3 << p.pos.z;
		std::string strObj3 = streamObj3.str();
		a.append(strObj3);

		a.append("),");
	}
	a.append("]");

	return a;
}

void ParticleController::getVectors(){
	vectors = (float2*)malloc(sizeof(float2)*nParticles);
	for (int i = 0; i < nParticles; i++){
		vectors[i] = make_float2(particles[i].pos.x, particles[i].pos.y);
	}
}

__device__
void ParticleController::cuda_getVectors(){
	vectors = (float2*)malloc(sizeof(float2)*nParticles);
	for (int i = 0; i < nParticles; i++){
		vectors[i] = make_float2(particles[i].pos.x, particles[i].pos.y);
	}
}

void ParticleController::update() {
	for (int pi = 0; pi < nParticles; pi++)
	{
		//Begin Loop 1
		Particle &p = particles[pi];
		Material& mat = *p.mat;

		float gu = 0, gv = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *ppy = p.py;
		float* pgx = p.gx;
		float* pgy = p.gy;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				gu += phi * n->u2;
				gv += phi * n->v2;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				//Velocity gradient
				dudx += n->u2 * gx;
				dudy += n->u2 * gy;
				dvdx += n->v2 * gx;
				dvdy += n->v2 * gy;
			}
		}

		//Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		p.T00 += .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);

		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		p.x += gu;
		p.y += gv;

		p.gu = gu;
		p.gv = gv;

		p.u += mat.smoothing*(gu - p.u);
		p.v += mat.smoothing*(gv - p.v);

		//Hard boundary correction (Random numbers keep it from clustering)
		if (p.x < 1) {
			p.x = 1 + .05;
		}
		else if (p.x > gSizeX - 2) {
			p.x = gSizeX - 2 - .05;
		}
		if (p.y < 1) {
			p.y = 1 + .05;
		}
		else if (p.y > gSizeY - 2) {
			p.y = gSizeY - 2 - .05;
		}

		//Update grid cell index and kernel weights
		int cx = p.cx = (int)(p.x - .5f);
		int cy = p.cy = (int)(p.y - .5f);
		p.gi = cx * gSizeY + cy;

		float x = cx - p.x;
		float y = cy - p.y;

		//Quadratic interpolation kernel weights - Not meant to be changed
		ppx[0] = .5f * x * x + 1.5f * x + 1.125f;
		pgx[0] = x + 1.5f;
		x++;
		ppx[1] = -x * x + .75f;
		pgx[1] = -2 * x;
		x++;
		ppx[2] = .5f * x * x - 1.5f * x + 1.125f;
		pgx[2] = x - 1.5f;

		ppy[0] = .5f * y * y + 1.5f * y + 1.125f;
		pgy[0] = y + 1.5f;
		y++;
		ppy[1] = -y * y + .75f;
		pgy[1] = -2 * y;
		y++;
		ppy[2] = .5f * y * y - 1.5f * y + 1.125f;
		pgy[2] = y - 1.5f;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;
		int mi = p.mat->materialIndex;
		float *px = p.px;
		float *gx = p.gx;
		float *py = p.py;
		float *gy = p.gy;
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			float gxi = gx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float gyj = gy[j];
				float phi = pxi * pyj;
				//Add particle mass, velocity and density gradient to grid
				n->mass += phi * m;
				n->particleDensity += phi;
				n->u += phi * mu;
				n->v += phi * mv;
				n->cgx[mi] += gxi * pyj;
				n->cgy[mi] += pxi * gyj;
				n->active = true;
			}
		}
	}

	//Add active nodes to list
	int gSizeXY = gSizeX * gSizeY;
	active = (bool*)malloc(sizeof(bool)*gSizeXY);
	for (int i = 0; i < gSizeXY; i++) {
		Node& n = grid[i];
		if (n.active && n.mass > 0) {
			active[i] = true;
			n.active = false;
			n.ax = n.ay = 0;
			n.gx = 0;
			n.gy = 0;
			n.u /= n.mass;
			n.v /= n.mass;
			for (int j = 0; j < numMaterials; j++) {
				n.gx += n.cgx[j];
				n.gy += n.cgy[j];
			}
			for (int j = 0; j < numMaterials; j++) {
				n.cgx[j] -= n.gx - n.cgx[j];
				n.cgy[j] -= n.gy - n.cgy[j];
			}
		}
		else{
			active[i] = false;
		}
	}

	// Calculate pressure and add forces to grid
	for (int pi = 0; pi < nParticles; pi++)
	{
		Particle& p = particles[pi];
		Material& mat = *p.mat;

		float fx = 0, fy = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0, sx = 0, sy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *pgx = p.gx;
		float *ppy = p.py;
		float *pgy = p.gy;

		int materialId = mat.materialIndex;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u * gx;
				dudy += n->u * gy;
				dvdx += n->v * gx;
				dvdy += n->v * gy;

				// Surface tension
				sx += phi * n->cgx[materialId];
				sy += phi * n->cgy[materialId];
			}
		}

		int cx = (int)p.x;
		int cy = (int)p.y;
		int gi = cx * gSizeY + cy;

		Node& n1 = grid[gi];
		Node& n2 = grid[gi + 1];
		Node& n3 = grid[gi + gSizeY];
		Node& n4 = grid[gi + gSizeY + 1];
		float density = uscip(n1.particleDensity, n1.gx, n1.gy, n2.particleDensity, n2.gx, n2.gy, n3.particleDensity, n3.gx, n3.gy, n4.particleDensity, n4.gx, n4.gy, p.x - cx, p.y - cy);

		float pressure = mat.stiffness / mat.restDensity * (density - mat.restDensity);
		if (pressure > 2) {
			pressure = 2;
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		D00 -= trace;
		D11 -= trace;
		p.T00 += .5f * (-wT0 + D00 - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + D11 - mat.meltRate * p.T11);

		// Stress tensor fracture
		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		float T00 = mat.mass * (mat.kElastic * p.T00 + mat.viscosity * D00 + pressure + trace * mat.bulkViscosity);
		float T01 = mat.mass * (mat.kElastic * p.T01 + mat.viscosity * D01);
		float T11 = mat.mass * (mat.kElastic * p.T11 + mat.viscosity * D11 + pressure + trace * mat.bulkViscosity);

		// Surface tension
		float lenSq = sx * sx + sy * sy;
		if (lenSq > 0)
		{
			float len = sqrtf(lenSq);
			float a = mat.mass * mat.surfaceTension / len;
			T00 -= a * (.5f * lenSq - sx * sx);
			T01 -= a * (-sx * sy);
			T11 -= a * (.5f * lenSq - sy * sy);
		}

		// Wall force
		if (p.x < 4) {
			fx += (4 - p.x);
		}
		else if (p.x > gSizeX - 5) {
			fx += (gSizeX - 5 - p.x);
		}
		if (p.y < 4) {
			fy += (4 - p.y);
		}
		else if (p.y > gSizeY - 5) {
			fy += (gSizeY - 5 - p.y);
		}


		// Add forces to grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;

				float gx = gxi * pyj;
				float gy = pxi * gyj;
				n->ax += -(gx * T00 + gy * T01) + fx * phi;
				n->ay += -(gx * T01 + gy * T11) + fy * phi;
			}
		}

		//Assign final particle Position
		p.pos.x = p.x*scale;
		p.pos.y = p.y*scale;
		p.trail.x = (p.x - p.gu)*scale;
		p.trail.y = (p.y - p.gv)*scale;

	}

	//Update acceleration of nodes
	for (int i = 0; i < gSizeXY; i++)
	{
		if (active[i]){
			Node& n = grid[i];
			n.u2 = 0;
			n.v2 = 0;
			n.ax /= n.mass;
			n.ay /= n.mass;
		}
	}

	for (int pi = 0; pi < nParticles; pi++)
	{
		Particle& p = particles[pi];
		Material& mat = *p.mat;
		// Update particle velocities
		Node* n = &grid[p.gi];
		float *px = p.px;
		float *py = p.py;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				p.u += phi * n->ax;
				p.v += phi * n->ay;
			}
		}

		p.v += mat.gravity;
		p.u *= 1 - mat.damping;
		p.v *= 1 - mat.damping;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;

		// Add particle velocities back to the grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				n->u2 += phi * mu;
				n->v2 += phi * mv;
			}
		}
	}

	//Update node velocities
	for (int i = 0; i < gSizeXY; i++)
	{
		if (active[i]){
			Node& n = grid[i];
			n.u2 /= n.mass;
			n.v2 /= n.mass;

			n.mass = 0;
			n.particleDensity = 0;
			n.u = 0;
			n.v = 0;
			memset(n.cgx, 0, 2 * numMaterials * sizeof(float));
		}
	}
}

__device__ 
void ParticleController::cuda_update(){

}

__device__
float ParticleController::cuda_uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v)
{
	float dx = x00 - x01;
	float dy = y00 - y10;
	float a = p01 - p00;
	float b = p11 - p10 - a;
	float c = p10 - p00;
	float d = y11 - y01;
	return ((((d - 2 * b - dy) * u - 2 * a + y00 + y01) * v +
		((3 * b + 2 * dy - d) * u + 3 * a - 2 * y00 - y01)) * v +
		((((2 * c - x00 - x10) * u + (3 * b + 2 * dx + x10 - x11)) * u - b - dy - dx) * u + y00)) * v +
		(((x11 - 2 * (p11 - p01 + c) + x10 + x00 + x01) * u +
		(3 * c - 2 * x00 - x10)) * u +
		x00) * u + p00;
}

__device__
void ParticleController::PreparePrint(string text){
	
}

__device__
void ParticleController::device_initializeGrid(int sizeX, int sizeY) {
	gSizeX = sizeX;
	gSizeY = sizeY;
	gSizeY_3 = sizeY - 3;
	grid = new Node[gSizeX*gSizeY];
	for (int i = 0; i < gSizeX*gSizeY; i++) {
		grid[i] = Node();
	}
}

__device__
void ParticleController::device_addParticles() {
	 nParticles = 300;
	 int p_index = 0;

	 particles = (Particle*)malloc(sizeof(Particle)*nParticles);
	 // Material 1
	 for (int i = 0; i < 10; i++) {
	 	for (int j = 0; j < 10; j++) {
	 		Particle p(&materials[0], i*.4 + 6, j*.8 / 5 + 6, make_float4(1, 0.5, 0.5, 1));
	 		p.initializeWeights(gSizeY);
	 		particles[p_index] = p;
	 		p_index++;
	 	}
	 }

	 // Material 2
	 for (int i = 0; i < 20; i++) {
	 	for (int j = 0; j < 5; j++) {
	 		Particle p(&materials[1], i*.4 + 150, j*.8 / 5 + 15, make_float4(1, 1, 1, 1));
	 		p.initializeWeights(gSizeY);
	 		particles[p_index] = p;
	 		p_index++;
	 	}
	 }

	 // Material 2
	 for (int i = 0; i < 10; i++) {
	 	for (int j = 0; j < 10; j++) {
	 		Particle p(&materials[2], i*.4 + 350, j*.8 / 5 + 15, make_float4(0.5, 1.0, 0.0, 1));
	 		p.initializeWeights(gSizeY);
	 		particles[p_index] = p;
	 		p_index++;
	 	}
	 }

	 P_Materials = (Material*)malloc(nParticles*sizeof(Material));
	 for (int i = 0; i < nParticles; i++){
	 	P_Materials[i] = *(particles[i].mat);
	 }

}
	