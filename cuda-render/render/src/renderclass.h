#ifndef RENDER_CLASS_H_
#define RENDER_CLASS_H_

#include <iostream>
using namespace std;

#include <vector>
#include <string>

// Include GLEW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

////////////////////////////////////////////////////////////
// mesh object
typedef struct {
	string mtlname = "";
	float Ka[3] = { 0.0f, 0.0f, 0.0f };
	float Kd[3] = { 1.0f, 1.0f, 1.0f };
	float Ks[3] = { 0.0f, 0.0f, 0.0f };
	int Ns = 10;
	int illum = 2;
	string map_Kd = "";
	bool istex = false;
} material;

typedef struct {
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> uvs;
	bool istex = false;
	int texid = -1;
} group;

typedef struct {
	vector<glm::vec3> vertices;
	vector<group> meshes;
	vector<material> mas;
	vector<cv::Mat> ims;
	int texnum = 0;
} mesh;

// render
class render {
public:
	render(int height, int width, int maxsz, double reduceratio = 0.8) {
		this->height = height;
		this->width = width;
		this->maxsz = maxsz;
		this->reduceratio = reduceratio;

		initialGLFW();
		initialGLEW();
		initialGL();
	}

	~render() {

		delete[] result;
		delete[] result3;

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		// unregister this buffer object with CUDA
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &tex_screen);
		glDeleteTextures(1, &tex_saver);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDeleteFramebuffers(1, &fbo_obj);

		// Use our shader
		glUseProgram(0);
		glDeleteProgram(programID);

		// Close OpenGL window and terminate GLFW
		glfwTerminate();
	}

public:
	// window
	GLFWwindow *window;
	int height;
	int width;
	int maxsz;
	double reduceratio;

	// Initialize GLFW & GLEW
	bool initialGLFW();
	bool initialGLEW();
	bool initialGL();

public:
	// fbo & big tex
	GLuint fbo_obj;
	GLuint tex_screen;
	GLuint tex_saver;
	int maxwidth;
	int tid;

	// cuda
	struct cudaGraphicsResource *cuda_tex_saver_resource;
	float *cuda_dest_saver_resource;
	float *result;

	float *cuda_dest_saver_resource3;
	float *result3;

	// void createfbo(GLuint &fbo, GLuint &tex, int tid, int h, int w);
	// void createbigtex();
	void initializecuda();
	void initializecuda2();

public:
	// program
	GLuint programID;

	GLuint ModelMatrixID;
	GLuint ViewMatrixID;

	GLuint lightPosition_modelspace;

	GLuint MaterialAmbient;
	GLuint MaterialDiffuse;
	GLuint MaterialSpecular;
	GLuint Shininess;

	GLuint TextureID;
	GLuint istex;
	GLuint isconfocal;
	GLuint isprojection;
	GLuint isalbedo;

	// shader
	GLuint LoadShaders(const char *vertex_file_path,
			const char *fragment_file_path);

	void programobj();

public:
	mesh loadobj(string folder, string name, bool &suc);

	int objnums;
	int texnums;
	vector<GLuint> VAOs;
	vector<GLuint> VBOs;
	vector<GLuint> normals;
	vector<GLuint> uvs;
	vector<GLuint> textures;

	void loadmesh(mesh obj);
	void drawmesh(mesh obj);
	void deletemesh();

public:
	// camera
	/////////////////////////////////////////////////////
	std::vector<glm::mat4> getViewMatrix(int N, double ratio);

	glm::mat4 getProjectionMatrix();

	glm::mat4 getModelMatrix(float _x, float _y, float _z, float xshift,
			float yshift, float zshift);

	// render params
	void display(string svfolder, mesh obj, int shininesslevel, int sz,
			int rnum, int lighthnum, int lightvnum, int hnum, int vnum,
			bool inirotshift = false, float rotx = 0, float roty = 0,
			float rotz = 0, float shiftx = 0, float shifty = 0,
			float shiftz = 0);	//, vector<string>rots, vector<string> shifts);
};

////////////////////////////////////////////////////////

#endif

