//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;

	Dnum(float f0 = 0, T d0 = T(0)) {
		f = f0, d = d0;
	}

	Dnum operator+(Dnum r) {
		return Dnum(f + r.f, d + r.d);
	}

	Dnum operator-(Dnum r) {
		return Dnum(f - r.f, d - r.d);
	}

	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}

	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }

typedef Dnum<vec2> Dnum2;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * M_PI / 180.0f;
		fp = 1;
		bp = 400;
	}

	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		mat4 M = mat4(
			u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1
		);

		return TranslateMatrix(wEye * (-1)) * M;
	}

	mat4 P() {
		return mat4(
			1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0
		);
	}

	void Animate() {
		vec3 d = wEye - wLookat;
		wEye = vec3(d.x * cosf(0.01f) + d.y * sinf(0.01f), -d.x * sinf(0.01f) + d.y * cosf(0.01f), d.z) + wLookat;
	}
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

struct Material {
	vec3 ka, kd, ks;
	float shininess;
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;
 
		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;
		layout(location = 2) in vec2 vtxUV;
 
		out	vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec4 vertexPos;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			
			for(int i = 0; i < nLights; i++)
			{
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
 
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			vertexPos = vec4(vtxPos, 1);	
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;
		
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 ka, ks, kd;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
 
		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		in vec4 vertexPos;
 
		out vec4 fragmentColor;
 
		void main(){
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);			
			if(dot(N, V) < 0)
			{
				N = -N;
			}
 
			vec3 ka = material.ka;
			vec3 kd = material.kd;
			vec3 ks = material.ks;
			float shininess = material.shininess;
 
			vec3 radiance = vec3(0, 0, 0);
	
			for(int i = 0; i < nLights; i++)
			{
				vec3 LeIn = lights[i].Le / dot(lights[i].wLightPos - vertexPos, lights[i].wLightPos - vertexPos);
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cosTheta = max(dot(N, L), 0);
				float cosDelta = max(dot(N, H), 0);
				radiance += ka * lights[i].La + (kd * cosTheta + ks * pow(cosDelta, shininess)) * LeIn;
			}
 
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (int i = 0; i < state.lights.size(); i++)
		{
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	virtual void Draw() = 0;

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

const int tesselationLevel = 200;

class ParamSurface : public Geometry {
	unsigned int nVtxStrip, nStrips;
public:
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	ParamSurface() {
		nVtxStrip = nStrips = 0;
	}

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vertexData;
		vertexData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0));
		Dnum2 V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vertexData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x);
		vec3 drdV(X.d.y, Y.d.y, Z.d.y);
		vertexData.normal = cross(drdU, drdV);
		return vertexData;
	}

	void Create(int N = tesselationLevel, int M = tesselationLevel) {
		nVtxStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j <= M; j++)
			{
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxStrip, nVtxStrip);
		}
	}
};

struct Sphere : public ParamSurface {
	float radius;

	Sphere(float _radius) {
		radius = _radius;
		Create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		V = V * M_PI;
		X = Cos(U) * Sin(V) * 2.0f;
		Y = Sin(U) * Sin(V) * 2.0f;
		Z = Cos(V) * 2.0f;
	}
};

struct Cylinder : public ParamSurface {
	float radius;
	float height;

	Cylinder(float _radius, float _height) {
		radius = _radius;
		height = _height;
		Create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = Cos(U) * radius;
		Y = Sin(U) * radius;
		Z = V * height;
	}
};

struct Paraboloid : public ParamSurface {
	Paraboloid() {
		Create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 3.0f;
		V = V * 2.0f * M_PI;
		X = U * Cos(V) * 2.75f;
		Y = U * Sin(V) * 2.75f;
		Z = U * U;
	}
};

struct CylinderPlane : public ParamSurface {
	float radius;

	CylinderPlane(float _radius) {
		radius = _radius;
		Create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = V * Cos(U) * radius;
		Y = V * Sin(U) * radius;
		Z = 0;
	}
};

struct Plane : public ParamSurface {
	Plane() {
		Create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		V = V * 2.0f * M_PI;
		X = Cos(U) * 1000.0f + Sin(U) * 1000.0f;
		Y = Cos(V) * 1000.0f + Sin(V) * 1000.0f;
		Z = 0;
	}
};

std::vector<Light> lights;

struct Lamp {
	Plane* plane;
	CylinderPlane* cylinderPlane;
	Cylinder* foot, * arm1, * arm2;
	Sphere* joint1, * joint2, * joint3;
	Paraboloid* head;
	Material* armMaterial, * jointMaterial, * planeMaterial;
	Shader* shader;

	vec3 translation, rotationAxis;
	float rotationAngle;

	Lamp() : translation(vec3(0, 0, 0)), rotationAxis(vec3(0, 0, 1)), rotationAngle(0) {
		SetupMaterials();
		Buil();
	}

	void SetupMaterials() {
		planeMaterial = new Material;
		planeMaterial->kd = vec3(0.3f, 0.2f, 0.1f);
		planeMaterial->ks = vec3(2, 2, 2);
		planeMaterial->ka = planeMaterial->kd * M_PI;
		planeMaterial->shininess = 50;

		jointMaterial = new Material;
		jointMaterial->kd = vec3(0.4f, 0.4f, 0.1f);
		jointMaterial->ks = vec3(2, 2, 2);
		jointMaterial->ka = jointMaterial->kd * M_PI;
		jointMaterial->shininess = 50;

		armMaterial = new Material;
		armMaterial->kd = vec3(0.22f, 0, 0);
		armMaterial->ks = vec3(2, 2, 2);
		armMaterial->ka = armMaterial->kd * M_PI;
		armMaterial->shininess = 50;
	}

	void Buil() {
		shader = new PhongShader();
		plane = new Plane();
		cylinderPlane = new CylinderPlane(10);
		foot = new Cylinder(10, 1.5f);
		joint1 = new Sphere(2);
		arm1 = new Cylinder(1, 15);
		joint2 = new Sphere(2);
		arm2 = new Cylinder(1, 15);
		joint3 = new Sphere(2);
		head = new Paraboloid();
	}

	void DrawJoint1(mat4& M, mat4& Minv) {
		rotationAxis = vec3(0.8f, 0.5f, 3);
		M = RotationMatrix(rotationAngle, rotationAxis);
		Minv = RotationMatrix(-rotationAngle, rotationAxis);
	}

	void DrawArm1(mat4& M, mat4& Minv) {
		M = RotationMatrix(rotationAngle, rotationAxis) * M;
		Minv = Minv * RotationMatrix(-rotationAngle, rotationAxis);
	}

	void DrawJoint2(mat4& M, mat4& Minv) {
		rotationAxis = vec3(1, 2, 3);
		M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(vec3(0, 0, 15)) * M;
		Minv = Minv * TranslateMatrix(-vec3(0, 0, 15)) * RotationMatrix(-rotationAngle, rotationAxis);
	}

	void DrawArm2(mat4& M, mat4& Minv) {
		M = RotationMatrix(rotationAngle, rotationAxis) * M;
		Minv = Minv * RotationMatrix(-rotationAngle, rotationAxis);
	}

	void DrawJoint3(mat4& M, mat4& Minv) {
		rotationAxis = vec3(1, -3, 5);
		M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(vec3(0, 0, 15)) * M;
		Minv = Minv * TranslateMatrix(-vec3(0, 0, 15)) * RotationMatrix(-rotationAngle, rotationAxis);
	}

	void DrawHead(mat4& M, mat4& Minv) {
		M = RotationMatrix(rotationAngle, rotationAxis) * M;
		Minv = Minv * RotationMatrix(-rotationAngle, rotationAxis);
	}

	void SetLightTransform(mat4& M, mat4& Minv) {
		M = TranslateMatrix(vec3(0, 0, 4)) * M;
		Minv = Minv * TranslateMatrix(vec3(0, 0, -4));
	}

	void Draw(RenderState state) {
		state.M = TranslateMatrix(vec3(0, 0, -1.5f));
		state.Minv = TranslateMatrix(vec3(0, 0, 1.5f));
		state.MVP = state.M * state.V * state.P;
		state.material = planeMaterial;
		shader->Bind(state);
		plane->Draw();

		state.M = TranslateMatrix(vec3(0, 0, -1.5f));
		state.Minv = TranslateMatrix(vec3(0, 0, 1.5f));
		state.MVP = state.M * state.V * state.P;
		state.material = armMaterial;
		shader->Bind(state);
		foot->Draw();

		state.M = TranslateMatrix(vec3(0, 0, 0));
		state.Minv = TranslateMatrix(vec3(0, 0, 0));
		state.MVP = state.M * state.V * state.P;
		state.material = armMaterial;
		shader->Bind(state);
		cylinderPlane->Draw();

		mat4 M, Minv;

		DrawJoint1(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = jointMaterial;
		shader->Bind(state);
		joint1->Draw();

		DrawArm1(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = armMaterial;
		shader->Bind(state);
		arm1->Draw();

		DrawJoint2(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = jointMaterial;
		shader->Bind(state);
		joint2->Draw();

		DrawArm2(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = armMaterial;
		shader->Bind(state);
		arm2->Draw();

		DrawJoint3(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = jointMaterial;
		shader->Bind(state);
		joint3->Draw();

		DrawHead(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = armMaterial;
		shader->Bind(state);
		head->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		rotationAngle = 1.2f * tend;
	}
};

Camera camera;

class Scene {
	Lamp* lamp;
public:
	void Build() {
		camera.wEye = vec3(0, 75, 40);
		camera.wLookat = vec3(0, 0, 15);
		camera.wVup = vec3(0, 0, 1);

		Light pointLight;
		pointLight.wLightPos = vec4(-15, 20, 10, 0);
		pointLight.La = vec3(0.15f, 0.15f, 0.15f);
		pointLight.Le = vec3(300, 300, 300);
		lights.push_back(pointLight);

		lamp = new Lamp();
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		lamp->Draw(state);
	}

	void Animate(float tstart, float tend) {
		lamp->Animate(tstart, tend);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.15f, 0.15f, 0.15f, 0.15f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = 0; t < tend; t += dt)
	{
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	camera.Animate();
	glutPostRedisplay();
}