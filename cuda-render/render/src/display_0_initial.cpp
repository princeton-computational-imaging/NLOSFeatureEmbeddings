#include "renderclass.h"

// Initialize GLFW
bool render::initialGLFW() {
	if (!glfwInit()) {
		cout << "Failed to initialize GLFW." << endl;
		getchar();
		return false;
	}

	// we use opengl 4.2
	// glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// osmesa
	glfwWindowHint(GLFW_VISIBLE, 0);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(width, height, "Tutorial 08 - Basic Shading",
	NULL,
	NULL);
	if (window == NULL) {
		cout << "Failed to open GLFW window. "
				"If you have an Intel GPU, they are not 3.3 compatible. "
				"Try the 2.1 version of the tutorials." << endl;
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);
	std::cout << "initialGLFW error\t" << glGetError() << std::endl;
	return true;
}

// Initialize GLEW
bool render::initialGLEW() {

	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		cout << "Failed to initialize GLEW." << endl;
		glfwTerminate();
		return false;
	}

	std::cout << "initialGLEW error\t" << glGetError() << std::endl;
	return true;
}

bool render::initialGL() {
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	// glDisable(GL_DEPTH_TEST);

	glDisable(GL_BLEND);
	// glEnable(GL_BLEND);
	// glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO,
	// GL_ONE);

	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	std::cout << "initialGL error\t" << glGetError() << std::endl;
	return true;
}

