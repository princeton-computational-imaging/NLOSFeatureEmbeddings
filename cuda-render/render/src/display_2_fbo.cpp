#include "renderclass.h"

///////////////////////////////////////////////////////////////////////
void createfbo(GLuint &fbo, GLuint &tex, int tid, int hei, int wid) {
	// GLuint fbo = 0;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// create a texture
	// GLuint tex_screen;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0 + tid);
	glBindTexture(GL_TEXTURE_2D, tex);

	/*
	 glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, wid, hei, 0, GL_RGB,
	 GL_UNSIGNED_BYTE, NULL);

	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
	 GL_LINEAR_MIPMAP_LINEAR);

	 // ... which requires mipmaps. Generate them automatically.
	 glGenerateMipmap(GL_TEXTURE_2D);
	 */

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, wid, hei, 0, GL_RGBA,
	GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
			tex, 0);

	// depth
	GLuint rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, wid, hei);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
	GL_RENDERBUFFER, rbo);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		cout << "ERROR::FRAMEBUFFER::"
				"Framebuffer is not complete!" << endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	cout << "fbo\t" << fbo << "\trbo\t" << rbo << "\ttex_screen\t" << tex
			<< "\ttex_act\t" << tid << endl;

	std::cout << "createfbo error\t" << glGetError() << std::endl;
}

void createbigtex(GLuint &tex, int tid, int hei, int wid, int maxsz,
		int &maxwidth) {
	// GLuint tex_screen;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0 + tid);
	glBindTexture(GL_TEXTURE_2D, tex);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	///////////////////////////////////////////////////////
	assert(hei == wid);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxwidth);
	cout << "maxlen\t" << maxwidth << endl;
	if (maxsz * wid > maxwidth) {
		cout << "maxsz * width\t" << maxsz * wid << endl;
		cout << "too big maxsz" << endl;
		exit(1);
	}

	maxwidth = wid * maxsz;
	cout << "use\t" << maxwidth << endl;

	///////////////////////////////////////////////////////////
	/*
	 glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, maxwidth, maxwidth, 0, GL_RGB,
	 GL_UNSIGNED_BYTE, NULL);

	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
	 GL_LINEAR_MIPMAP_LINEAR);

	 // ... which requires mipmaps. Generate them automatically.
	 glGenerateMipmap(GL_TEXTURE_2D);
	 */

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, maxwidth, maxwidth, 0, GL_RGBA,
	GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
	// GL_LINEAR_MIPMAP_LINEAR);

	// ... which requires mipmaps. Generate them automatically.
	// glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	std::cout << "createbigtex error\t" << glGetError() << std::endl;
}

void render::initializecuda() {
	// create fbo, using first and second texture
	// GLuint fbo_obj, tex_screen;
	tid = 0;
	createfbo(fbo_obj, tex_screen, tid, height, width);

	tid = 1;
	createbigtex(tex_saver, tid, height, width, maxsz, maxwidth);

	tid = 2;

	// cuda
	initializecuda2();
}

