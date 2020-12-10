#include "renderclass.h"

glm::mat4 render::getProjectionMatrix() {

	// Projection matrix : 45��� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	// ProjectionMatrix = glm::perspective(glm::radians(45.0f),
	// 	1.0f * width / height, 0.1f, 100.0f);
	glm::mat4 ProjectionMatrix = glm::ortho(-1.0, 1.0, -1.0, 1.0, -100.0,
			100.0);

	return ProjectionMatrix;
}

std::vector<glm::mat4> render::getViewMatrix(int samplenum, double ratio) {

	float pi = 3.141592653589793238463f;

	glm::vec3 zaxis(0.0f, 0.0f, 1.0f);

	int sam2 = static_cast<int>(1.0 * samplenum / ratio / ratio);
	int sambe = sam2 - samplenum;

	std::vector<glm::mat4> re;
	re.resize(samplenum);

	for (int i = 0; i < samplenum; i++) {

		float n = sambe + i + 1.0f;
		float N = sam2 + 1.0f;

		// large than 0, small than 1
		float zn = 1.0f * n / N;
		float r = std::sqrt(1.0f - zn * zn);

		float phi = (sqrt(5.0f) - 1.0f) / 2.0f;
		float angle = 2.0f * pi * n * phi;
		float xn = r * cos(angle);
		float yn = r * sin(angle);

		glm::vec3 newaxis(xn, yn, zn);

		float costheta = zn;
		float theta = glm::acos(costheta);

		glm::vec3 rotaxis = glm::cross(zaxis, newaxis);
		float rotaxislen = std::sqrt(
				rotaxis.x * rotaxis.x + rotaxis.y * rotaxis.y
						+ rotaxis.z * rotaxis.z);
		if (rotaxislen > 1e-5)
			rotaxis /= rotaxislen;
		else {
			assert(theta == 0.0f);
			std::cout << "rotate axis\t" << newaxis.x << "\t" << newaxis.y
					<< std::endl;
		}

		cv::Mat rotvec = cv::Mat::zeros(3, 1, CV_32FC1);
		rotvec.at<float>(0, 0) = theta * rotaxis.x;
		rotvec.at<float>(1, 0) = theta * rotaxis.y;
		rotvec.at<float>(2, 0) = theta * rotaxis.z;

		cv::Mat rotmtx;
		cv::Rodrigues(rotvec, rotmtx);

		re[i] = glm::mat4(1.0);
		re[i][0][0] = rotmtx.at<float>(0, 0);
		re[i][0][1] = rotmtx.at<float>(1, 0);
		re[i][0][2] = rotmtx.at<float>(2, 0);

		re[i][1][0] = rotmtx.at<float>(0, 1);
		re[i][1][1] = rotmtx.at<float>(1, 1);
		re[i][1][2] = rotmtx.at<float>(2, 1);

		re[i][2][0] = rotmtx.at<float>(0, 2);
		re[i][2][1] = rotmtx.at<float>(1, 2);
		re[i][2][2] = rotmtx.at<float>(2, 2);
	}

	return re;
}

glm::mat4 render::getModelMatrix(float _x, float _y, float _z, float xshift,
		float yshift, float zshift) {

	glm::mat4 Modelx = glm::rotate(glm::mat4(1.0), glm::radians(_x),
			glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 Modely = glm::rotate(glm::mat4(1.0), glm::radians(_y),
			glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 Modelz = glm::rotate(glm::mat4(1.0), glm::radians(_z),
			glm::vec3(0.0f, 0.0f, 1.0f));

	glm::mat4 Modelshift = glm::mat4(1.0);
	Modelshift[3] = glm::vec4(xshift, yshift, zshift, 1.0);

	glm::mat4 ModelMatirx = Modelshift * Modelz * Modely * Modelx;

	return ModelMatirx;
}

