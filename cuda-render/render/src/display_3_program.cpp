#include "renderclass.h"
#include <fstream>
#include <sstream>
#include <assert.h>

GLuint render::LoadShaders(const char *vertex_file_path,
		const char *fragment_file_path) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	} else {
		printf(
				"Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
				vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const *VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
				&VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const *FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
				&FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL,
				&ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}

//////////////////////////////////////////////////////////////////
void render::programobj() {
	//////////////////////////////////////////////////////////////////////
	// Create and compile our GLSL program from the shaders
	programID = LoadShaders("pointlight.vertexshader",
			"pointlight.fragmentshader");

	// first program
	glUseProgram(programID);

	/////////////////////////////////////////////////////////////////////
	// M & V will change!
	ModelMatrixID = glGetUniformLocation(programID, "M");
	ViewMatrixID = glGetUniformLocation(programID, "V");

	// P will not change!
	GLuint PersepctiveMatrixID = glGetUniformLocation(programID, "P");
	glm::mat4 ProjectionMatrix = getProjectionMatrix();
	glUniformMatrix4fv(PersepctiveMatrixID, 1,
	GL_FALSE, &ProjectionMatrix[0][0]);

	lightPosition_modelspace = glGetUniformLocation(programID,
			"lightPosition_modelspace");

	//////////////////////////////////////////////////////
	// it will change
	MaterialAmbient = glGetUniformLocation(programID, "MaterialAmbient");
	MaterialDiffuse = glGetUniformLocation(programID, "MaterialDiffuse");
	MaterialSpecular = glGetUniformLocation(programID, "MaterialSpecular");

	Shininess = glGetUniformLocation(programID, "Shininess");

	////////////////////////////////////////////////////////////////////////////
	// Get a handle for our "myTextureSampler" uniform
	// Set our "myTextureSampler" sampler to use Texture Unit 2
	TextureID = glGetUniformLocation(programID, "myTextureSampler");
	istex = glGetUniformLocation(programID, "istex");

	isprojection = glGetUniformLocation(programID, "isprojection");
	isconfocal = glGetUniformLocation(programID, "isconfocal");
	isalbedo = glGetUniformLocation(programID, "isalbedo");

	glUseProgram(0);
	std::cout << "programobj 1 error\t" << glGetError() << std::endl;

	return;
}

