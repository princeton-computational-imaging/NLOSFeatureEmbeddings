#include "renderclass.h"

void render::loadmesh(mesh obj) {
	objnums = obj.meshes.size();

	VAOs.resize(objnums);
	VBOs.resize(objnums);
	normals.resize(objnums);

	glGenVertexArrays(objnums, &VAOs[0]);
	glGenBuffers(objnums, &VBOs[0]);
	glGenBuffers(objnums, &normals[0]);

	texnums = obj.texnum;
	if (texnums > 0) {
		uvs.resize(texnums);
		textures.resize(texnums);

		glGenBuffers(texnums, &uvs[0]);
		glGenTextures(texnums, &textures[0]);

		for (int i = 0; i < texnums; i++) {
			cv::Mat img = obj.ims[i];
			int length = img.rows * img.cols * 3;
			uchar *buffer = new uchar[length];
			for (int ih = 0; ih < img.rows; ih++)
				for (int jw = 0; jw < img.cols; jw++)
					for (int kc = 0; kc < 3; kc++) {
						int idx = (img.rows - 1 - ih) * img.cols * 3 + jw * 3
								+ 2 - kc;
						if (img.type() == CV_8UC3)
							buffer[idx] = img.at<cv::Vec3b>(ih, jw)[kc];
						else
							buffer[idx] = img.at<uchar>(ih, jw);
					}

			// "Bind" the newly created texture : all future texture functions will modify this texture
			glActiveTexture(GL_TEXTURE0 + tid + i);

			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			// Give the image to OpenGL
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0,
			GL_RGB,
			GL_UNSIGNED_BYTE, buffer);

			// Poor filtering, or ...
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

			// ... nice trilinear filtering ...
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_LINEAR_MIPMAP_LINEAR);

			// ... which requires mipmaps. Generate them automatically.
			glGenerateMipmap(GL_TEXTURE_2D);

			free(buffer);
		}
	}

	for (int i = 0; i < objnums; i++) {
		glBindVertexArray(VAOs[i]);

		group tmp = obj.meshes[i];
		glBindBuffer(GL_ARRAY_BUFFER, VBOs[i]);
		glBufferData(GL_ARRAY_BUFFER, tmp.vertices.size() * sizeof(glm::vec3),
				&tmp.vertices[0], GL_STATIC_DRAW);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,                  // attribute
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized?
				0,                  // stride
				(void*) 0            // array buffer offset
				);

		glBindBuffer(GL_ARRAY_BUFFER, normals[i]);
		glBufferData(GL_ARRAY_BUFFER, tmp.normals.size() * sizeof(glm::vec3),
				&tmp.normals[0], GL_STATIC_DRAW);

		// 3rd attribute buffer : normals
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1,                                // attribute
				3,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*) 0                          // array buffer offset
				);

		if (tmp.istex) {
			glBindBuffer(GL_ARRAY_BUFFER, uvs[i]);
			glBufferData(GL_ARRAY_BUFFER, tmp.uvs.size() * sizeof(glm::vec2),
					&tmp.uvs[0],
					GL_STATIC_DRAW);

			// 2nd attribute buffer : UVs
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2,                                // attribute
					2,                                // size
					GL_FLOAT,                         // type
					GL_FALSE,                         // normalized?
					0,                                // stride
					(void*) 0                          // array buffer offset
					);
		}

	}
}

void render::deletemesh() {
	// glDisableVertexAttribArray(0);
	// glDisableVertexAttribArray(1);
	// glDisableVertexAttribArray(2);

	// Cleanup VBO and shader
	glDeleteBuffers(objnums, &VBOs[0]);
	glDeleteBuffers(objnums, &normals[0]);

	if (texnums > 0) {
		glDeleteBuffers(texnums, &uvs[0]);
		glDeleteTextures(texnums, &textures[0]);
	}

	glDeleteVertexArrays(objnums, &VAOs[0]);

	VBOs.clear();
	normals.clear();
	uvs.clear();
	textures.clear();
	VAOs.clear();
}

