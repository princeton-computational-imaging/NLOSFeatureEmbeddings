#include "renderclass.h"
#include <chrono>

#define TBE 0
#define TEN 6

////////////////////////////////////////////////////////
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
// #include <helper_cuda.h>
// #include <helper_cuda_gl.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/////////////////////////////////////////////////////////////////////////
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" void
launch_cudaProcess2(cudaArray *g_data_array, float *g_output_data, int timebin,
		int imgh, int imgw, int sz, int maxdepth, int mindepth);

extern "C" void
launch_cudaProcess3(cudaArray *g_data_array, float *g_output_data, int imgh,
		int imgw, int sz);

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
// initCUDA
// int argc = 1;
// void *argv;
// int deviceid = findCudaGLDevice(argc, (const char **) argv);
// cout << "device id" << deviceid << endl;
void render::initializecuda2() {

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	// cuda resources
	// struct cudaGraphicsResource *cuda_tex_saver_resource;

	// register this texture with CUDA
	CUDA_CHECK_RETURN(
			cudaGraphicsGLRegisterImage(&cuda_tex_saver_resource, tex_saver, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

	int timebin = (TEN - TBE) * 100;

	// set up vertex data parameter
	// float *cuda_dest_saver_resource;
	int size_tex_data = sizeof(float) * timebin * height * width * 3;
	CUDA_CHECK_RETURN(
			cudaMalloc((void ** ) &cuda_dest_saver_resource, size_tex_data));

	result = new float[timebin * width * height * 3];

	int size_tex_data3 = sizeof(float) * height * width * 3;
	CUDA_CHECK_RETURN(
			cudaMalloc((void ** ) &cuda_dest_saver_resource3, size_tex_data3));

	result3 = new float[width * height * 3];
}

////////////////////////////////////////////////////////////////////////////////
inline float randomnum() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

inline void savedistre(float *result_txhxwx3, float &maxval, float &maxdst,
		float & mindst, cv::Mat tmpsave, int height, int width, int tsz,
		char *fold, char *fname, int id) {

	for (int t = 0; t < tsz; t++)
		for (int ih = 0; ih < height; ih++)
			for (int jw = 0; jw < width; jw++)
				for (int kc = 0; kc < 3; kc++) {

					// opencv is BGR
					int idx = t * height * width * 3 + ih * width * 3 + jw * 3
							+ 2 - kc;

					float val = result_txhxwx3[idx];
					tmpsave.at<cv::Vec3f>(t * height + ih, jw)[kc] = val;

					// meaningful value
					if (val > 1e-8) {
						if (val > maxval)
							maxval = val;
						if (t > maxdst)
							maxdst = 1.0 * t / 100;
						if (t < mindst)
							mindst = 1.0 * t / 100;
					}
				}

	/*
	 *
	 // cv::imshow("tmp", tmpshow);
	 // cv::waitKey(0);

	 char name[256];
	 sprintf(name, "%s/%s-%d-%.4f-%.4f-%.4f.png", fold, fname, id, maxval,
	 maxdst, mindst);
	 cv::imwrite(name, tmpshow);


	 // Write to file!
	 std::vector<cv::Mat> rgbChannels(3);
	 cv::split(tmpsave, rgbChannels);
	 cv::Mat singlechannel = rgbChannels[0];
	 */
	char name[256];
	sprintf(name, "%s/%s-%d-%.4f-%.4f-%.4f.txt", fold, fname, id, maxval,
			maxdst, mindst);
	// cv::FileStorage fs(name, cv::FileStorage::WRITE);
	// fs << "mat1" << tmpsave;

	sprintf(name, "%s/%s-%d-%.4f-%.4f-%.4f.hdr", fold, fname, id, maxval,
			maxdst, mindst);
	cv::imwrite(name, tmpsave);
	return;
}

inline void savedistre3(float *result_hxwx3, cv::Mat tmpshow, cv::Mat tmpsave,
		int height, int width, int sz, char *fold, char *fname, int id) {

	float maxval = 0.0f;
	float minval = 100.0f;
	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++)
			for (int kc = 0; kc < 3; kc++) {

				// opencv is BGR
				int idx = ih * width * 3 + jw * 3 + 2 - kc;
				float val = result_hxwx3[idx] / sz / sz;
				tmpsave.at<cv::Vec3f>(ih, jw)[kc] = val;

				if (val > maxval)
					maxval = val;
				if (val < minval)
					minval = val;

				if (val > 1)
					val = 1;
				if (val < 0)
					val = 0;
				tmpshow.at<cv::Vec3b>(ih, jw)[kc] = (uchar) (255.0f * val);
			}

	// cv::imshow("tmp", tmpshow);
	// cv::waitKey(0);

	char name[256];
	sprintf(name, "%s/%s-%d-%.4f-%.4f.png", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpshow);

	/*
	 // Write to file!
	 std::vector<cv::Mat> rgbChannels(3);
	 cv::split(tmpsave, rgbChannels);
	 cv::Mat singlechannel = rgbChannels[0];
	 */

	// sprintf(name, "%s/%s-%d-%.4f-%.4f-%.4f.txt", fold, fname, id, maxval,
	//	maxdst, mindst);
	// cv::FileStorage fs(name, cv::FileStorage::WRITE);
	// fs << "mat1" << tmpsave;
	sprintf(name, "%s/%s-%d-%.4f-%.4f.hdr", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpsave);
	return;
}

inline void saveim(vector<float> tmpdata, cv::Mat tmpshow, cv::Mat tmpsave,
		int height, int width, char *fold, char *fname, int id) {

//after drawing
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, &tmpdata[0]);

	float maxval = 0.0f;
	float minval = 100.0f;
	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++)
			for (int kc = 0; kc < 3; kc++) {
				int idx = (height - 1 - ih) * width * 4 + jw * 4 + 2 - kc;

				bool lightpath = false;
				if (lightpath)
					idx = (height - 1 - ih) * width * 4 + jw * 4 + 3;

				float val = tmpdata[idx];
				tmpsave.at<cv::Vec3f>(ih, jw)[kc] = val;

				if (val > maxval)
					maxval = val;
				if (val < minval)
					minval = val;

				if (val > 1)
					val = 1;
				if (val < 0)
					val = 0;

				// tmpshow.at<cv::Vec3b>(ih, jw)[kc] = (uchar) (255.0f * val);
			}

// cv::imshow("tmp", tmpshow);
// cv::waitKey(0);

	char name[256];
	sprintf(name, "%s/%s-%d-%.4f-%.4f.png", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpshow);

// Write to file!
	// std::vector<cv::Mat> rgbChannels(3);
	// cv::split(tmpsave, rgbChannels);
	// cv::Mat singlechannel = rgbChannels[0];

	// sprintf(name, "%s/%s-%d-%.4f-%.4f.txt", fold, fname, id, maxval, minval);
	// cv::FileStorage fs(name, cv::FileStorage::WRITE);
	// fs << "mat1" << tmpsave;

	sprintf(name, "%s/%s-%d-%.4f-%.4f.hdr", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpsave);
	return;
}

inline void savedep(vector<float> tmpdata, cv::Mat tmpshow, cv::Mat tmpsave,
		int height, int width, char *fold, char *fname, int id) {

//after drawing
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,
			&tmpdata[0]);

	float maxval = -100.0f;
	float minval = 100.0f;
	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++) {

			int idx = (height - 1 - ih) * width + jw;

			float dis = tmpdata[idx];

			// not background
			if (abs(dis - 1) >= 1e-5) {
				dis = dis * 2 - 1;
				dis = -dis;
				dis = dis * 100;

				if (dis > maxval)
					maxval = dis;
				if (dis < minval)
					minval = dis;
			} else {
				dis = -1.0;
			}

			dis = (dis + 1) / 2;
			tmpsave.at<cv::Vec3f>(ih, jw) = cv::Vec3f(dis, dis, dis);

			// background
			float val = dis;
			if (val > 1)
				val = 1;
			if (val < 0)
				val = 0;

			uchar dis255 = (uchar) 255.0f * val;
			tmpshow.at<cv::Vec3b>(ih, jw) = cv::Vec3b(dis255, dis255, dis255);
		}

// cv::imshow("tmp", tmpshow);
// cv::waitKey(0);

	char name[256];
	sprintf(name, "%s/%s-%d-%.4f-%.4f.png", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpshow);

// Write to file!

	/*
	 // single channel
	 // Write to file!
	 std::vector<cv::Mat> rgbChannels(3);
	 cv::split(tmpsave, rgbChannels);
	 cv::Mat singlechannel = rgbChannels[0];

	 sprintf(name, "%s/%s-%d-%.4f-%.4f.txt", fold, fname, id, maxval, minval);
	 cv::FileStorage fs(name, cv::FileStorage::WRITE);
	 fs << "mat1" << singlechannel;
	 */

	sprintf(name, "%s/%s-%d-%.4f-%.4f.hdr", fold, fname, id, maxval, minval);
	cv::imwrite(name, tmpsave);

	return;
}

void render::display(string svfolder, mesh obj, int shininesslevel, int sz,
		int rnum, int lighthnum, int lightvnum, int hnum, int vnum,
		bool inirotshift, float rotx, float roty, float rotz, float shiftx,
		float shifty, float shiftz) {
	// ,vector<string> rots, vector<string> shifts) {

	// it cannot be large than maxsz, or it will exceed our max texture size
	assert(sz == maxsz);
	int texblockdim = static_cast<int>(std::ceil(1.0 * hnum / sz));
	int texthreaddim = sz;
	cout << "tblock dim\t" << texblockdim << endl;
	cout << "tthread dim\t" << texthreaddim << endl;

	////////////////////////////////////////////////////
	std::vector<float> tmpdata(width * height * 4);
	cv::Mat tmpshow = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat tmpsave = cv::Mat::zeros(height, width, CV_32FC3);

	float maxdst = 0.0, mindst = 100.0, maxval = 0.0, minval = 100.0;

	int timebin = (TEN - TBE) * 100;
	cv::Mat tmpshowall = cv::Mat::zeros(timebin * height, width, CV_8UC3);
	cv::Mat tmpsaveall = cv::Mat::zeros(timebin * height, width,
	CV_32FC3);

	// cuda set 0
	int size_tex_data = sizeof(float) * timebin * height * width * 3;
	// CUDA_CHECK_RETURN(cudaMemset(cuda_dest_saver_resource, 0, size_tex_data));
	// memset(result, 0, size_tex_data);

	int size_tex_data3 = sizeof(float) * height * width * 3;

//////////////////////////////////////////////////////////////////

	glm::mat4 I = glm::mat4(1.0f);
	vector<glm::mat4> views = getViewMatrix(hnum * vnum, reduceratio);

/////////////////////////////////////////////////////////////////////
// 3 levels
// 1 is low specular, 0-1
// 2 is midium, 1-256
// 3 is high, 256-512

// how many shininess we sample
// how many rotations translations we have
	for (int rid = 0; rid < rnum; rid++) {

		float xrot = 15 * (2 * randomnum() - 1);
		// float yrot = 180 * (2 * randomnum() - 1);
		float zrot = 15 * (2 * randomnum() - 1);

		float yrot = randomnum();
		if (yrot >= 0.5)
			yrot = 180 * yrot - 45; // [45, 135]
		else
			yrot = -180 * yrot - 45; // [-45, -135]

		// x, y move [-0.3, 0.3]
		// z move [0, 0.8]
		float xshift = 0.4f * (2 * randomnum() - 1);
		float yshift = 0.4f * (2 * randomnum() - 1);
		float zshift = 0.4f * (2 * randomnum() - 1);

		/*
		 float xrot = 15 * (2 * randomnum() - 1);
		 float yrot = 15 * (2 * randomnum() - 1);
		 float zrot = 180 * (2 * randomnum() - 1);

		 // x, y move [-0.3, 0.3]
		 float xshift = 0.4f * (2 * randomnum() - 1);
		 float yshift = 0.4f * (2 * randomnum() - 1);
		 float zshift = 0.4f * (2 * randomnum() - 1);
		 */

		if (inirotshift) {
			xrot = rotx;
			yrot = roty;
			zrot = rotz;
			xshift = shiftx;
			yshift = shifty;
			zshift = shiftz;
		}

		// shininess
		float shinessallllllll = 0;
		// float shinessallllllll = 32 + randomnum() * (512 - 32);

		glm::mat4 ModelMatrix = getModelMatrix(xrot, yrot, zrot, xshift, yshift,
				zshift);

		char folder[256];
		sprintf(folder, "%s/shine_%.4f-rot_%.4f_%.4f_%.4f-shift_%.4f_%.4f_%.4f",
				svfolder.c_str(), shinessallllllll, xrot, yrot, zrot, xshift,
				yshift, zshift);
		std::cout << folder << std::endl;

		char cmd[256];
		sprintf(cmd, "mkdir %s", folder);
		system(cmd);

		//////////////////////////////////////////////////////////
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
		glUseProgram(programID);

		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDisable(GL_BLEND);

		////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		// draw the semi sphere sampling
		int samplenum = (int) (100 * 0.5 * 0.5);
		vector<glm::mat4> views_orth = getViewMatrix(samplenum, 0.5);

		////////////////////////////////////////////////////////
		// merge light source

		glUniform1f(isprojection, -1.0f);
		glUniform1f(isalbedo, -1.0f);

		for (int ls = 0; ls < samplenum + 1; ls++) {

			if (ls < samplenum) {
				int renderid = ls;
				glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE,
						&views_orth[renderid][0][0]);
			} else {
				glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &I[0][0]);
			}

			int lnum = sz;
			assert(lnum <= sz);

			cudaMemset(cuda_dest_saver_resource3, 0, size_tex_data3);
			memset(result3, 0, size_tex_data3);

			for (int lh = 0; lh < lnum; lh++) {
				for (int lv = 0; lv < lnum; lv++) {

					float light_x = 2.0f * (lh + 1) / (lnum + 1) - 1.0f;
					float light_y = 2.0f * (lv + 1) / (lnum + 1) - 1.0f;
					float light_z = 1.0f;

					glUniform3f(lightPosition_modelspace, light_x, light_y,
							light_z);

					// Clear the screen
					glViewport(0, 0, width, height);
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

					for (int i = 0; i < objnums; i++) {

						glBindVertexArray(VAOs[i]);

						// draw an original image
						// default parameters
						float shiness = obj.mas[i].Ns;
						glUniform1f(Shininess, shiness);

						// glUniform3f(MaterialAmbient, 0.1f, 0.1f, 0.1f);
						glUniform3f(MaterialAmbient, obj.mas[i].Ka[0],
								obj.mas[i].Ka[1], obj.mas[i].Ka[2]);
						// glUniform3f(MaterialDiffuse, 1.0f, 1.0f, 1.0f);
						glUniform3f(MaterialDiffuse, obj.mas[i].Kd[0],
								obj.mas[i].Kd[1], obj.mas[i].Kd[2]);
						glUniform3f(MaterialSpecular, obj.mas[i].Ks[0],
								obj.mas[i].Ks[1], obj.mas[i].Ks[2]);

						if (obj.meshes[i].istex) {
							// Bind our texture in Texture Unit 0
							glActiveTexture(GL_TEXTURE0 + tid + i);
							glBindTexture(GL_TEXTURE_2D, textures[i]);

							// Set our "myTextureSampler" sampler to use Texture Unit 0
							glUniform1i(TextureID, tid + i);
							glUniform1f(istex, 1.0f);
						} else {
							glUniform1f(istex, -1.0f);
						}

						// Draw the triangles !
						glDrawArrays(GL_TRIANGLES, 0,
								obj.meshes[i].vertices.size());
					}

					int xshift = lh * width;
					int yshift = lv * height;

					// Bind our texture in Texture Unit 1
					glActiveTexture(GL_TEXTURE0 + tid - 1);
					glBindTexture(GL_TEXTURE_2D, tex_saver);
					glCopyTexSubImage2D(GL_TEXTURE_2D, 0, xshift, yshift, 0, 0,
							width, height);

				}
			}

			//////////////////////////////////
			// do we need this?
			glBindTexture(GL_TEXTURE_2D, 0);

			// map buffer objects to get CUDA device pointers
			CUDA_CHECK_RETURN(
					cudaGraphicsMapResources(1, &cuda_tex_saver_resource, 0));
			//printf("Mapping tex_in\n");

			// texture pointer
			cudaArray *in_array;
			CUDA_CHECK_RETURN(
					cudaGraphicsSubResourceGetMappedArray(&in_array,
							cuda_tex_saver_resource, 0, 0));
			// cout << static_cast<int>(in_array)<<endl;

			/////////////////////////////////////////////////////////////////////////////
			// execute CUDA kernel
			// cudaArray *g_data_array,	float *g_output_data, int timebin, int imgh, int imgw, int sz
			launch_cudaProcess3(in_array, cuda_dest_saver_resource3, height,
					width, lnum);
			////////////////////////////////////////////////////////////////////////////

			CUDA_CHECK_RETURN(
					cudaGraphicsUnmapResources(1, &cuda_tex_saver_resource, 0));

			cudaMemcpy(result3, cuda_dest_saver_resource3, size_tex_data3,
					cudaMemcpyDeviceToHost);

			savedistre3(result3, tmpshow, tmpsave, height, width, lnum, folder,
					"confocal", (ls + 1) % (samplenum + 1));

		}

		////////////////////////////////////////////////////////////
		glUniform3f(lightPosition_modelspace, 0.0f, 0.0f, 1.0f);

		glUniform1f(isprojection, -1.0f);
		glUniform1f(isalbedo, -1.0f);

		////////////////////////////////////////////////////////////////
		for (int ls = 0; ls < samplenum + 1; ls++) {

			if (ls < samplenum) {
				int renderid = ls;
				glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE,
						&views_orth[renderid][0][0]);
			} else {
				glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &I[0][0]);
			}

			// Clear the screen
			glViewport(0, 0, width, height);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			for (int i = 0; i < objnums; i++) {

				glBindVertexArray(VAOs[i]);

				// draw an original image
				// default parameters
				float shiness = obj.mas[i].Ns;
				glUniform1f(Shininess, shiness);

				// glUniform3f(MaterialAmbient, 0.1f, 0.1f, 0.1f);
				glUniform3f(MaterialAmbient, obj.mas[i].Ka[0], obj.mas[i].Ka[1],
						obj.mas[i].Ka[2]);
				// glUniform3f(MaterialDiffuse, 1.0f, 1.0f, 1.0f);
				glUniform3f(MaterialDiffuse, obj.mas[i].Kd[0], obj.mas[i].Kd[1],
						obj.mas[i].Kd[2]);
				glUniform3f(MaterialSpecular, obj.mas[i].Ks[0],
						obj.mas[i].Ks[1], obj.mas[i].Ks[2]);

				if (obj.meshes[i].istex) {
					// Bind our texture in Texture Unit 0
					glActiveTexture(GL_TEXTURE0 + tid + i);
					glBindTexture(GL_TEXTURE_2D, textures[i]);

					// Set our "myTextureSampler" sampler to use Texture Unit 0
					glUniform1i(TextureID, tid + i);
					glUniform1f(istex, 1.0f);
				} else {
					glUniform1f(istex, -1.0f);
				}

				// Draw the triangles !
				glDrawArrays(GL_TRIANGLES, 0, obj.meshes[i].vertices.size());
			}

			saveim(tmpdata, tmpshow, tmpsave, height, width, folder, "original",
					(1 + ls) % (samplenum + 1));

			// one depth
			savedep(tmpdata, tmpshow, tmpsave, height, width, folder, "depth",
					(1 + ls) % (samplenum + 1));
		}

		///////////////////////////////////////////////////////////////
		// draw combined image
		for (int conf = 1; conf < 2; conf++) {

			float confocal = conf * 2.0 - 1.0;
			glUniform1f(isconfocal, confocal);

			for (int lh = 0; lh < lighthnum; lh++) {
				for (int lv = 0; lv < lightvnum; lv++) {

					// memory
					cout << "light idx" << lh * lightvnum + lv << endl;
					cudaMemset(cuda_dest_saver_resource, 0, size_tex_data);
					memset(result, 0, size_tex_data);

					// draw
					cout << "light idx" << lh * lightvnum + lv << endl;
					if (lh * lightvnum + lv > 0)
						exit(0);
					float light_x = 2.0f * (lh + 1) / (lighthnum + 1) - 1.0f;
					float light_y = 2.0f * (lv + 1) / (lightvnum + 1) - 1.0f;
					float light_z = 1.0f;

					glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
					glUseProgram(programID);
					glUniform3f(lightPosition_modelspace, light_x, light_y,
							light_z);

					// record time
					auto start = std::chrono::high_resolution_clock::now();

					for (int hb = 0; hb < texblockdim; hb++) {
						for (int wb = 0; wb < texblockdim; wb++) {

							// cout << "block idx" << hb << "\t" << wb << endl;

							glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
							glUseProgram(programID);

							glUniform1f(isprojection, 1.0f);
							glUniform1f(isalbedo, -1.0f);

							glEnable(GL_DEPTH_TEST);
							glDepthFunc(GL_LESS);
							glDisable(GL_BLEND);

							glViewport(0, 0, width, height);

							for (int ht = 0; ht < texthreaddim; ht++) {

								int hidx = hb * sz + ht;
								if (hidx > hnum)
									continue;

								for (int wt = 0; wt < texthreaddim; wt++) {

									int widx = wb * sz + wt;
									if (widx > vnum)
										continue;

									int xshift = wt * width;
									int yshift = ht * height;

									int renderid = hb * texblockdim
											* texthreaddim * texthreaddim
											+ wb * texthreaddim * texthreaddim
											+ ht * texthreaddim + wt;

									glUniformMatrix4fv(ViewMatrixID, 1,
									GL_FALSE, &views[renderid][0][0]);

									// Clear the screen
									glClear(
									GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

									for (int i = 0; i < objnums; i++) {

										glBindVertexArray(VAOs[i]);

										// draw an original image
										float shiness = obj.mas[i].Ns;

										bool specular = false;
										if (specular) {
											shinessallllllll = 64;
											glUniform1f(Shininess,
													shinessallllllll); // shiness
											glUniform3f(MaterialDiffuse, 0.0f,
													0.0f, 0.0f);
										} else {
											glUniform1f(Shininess, shiness);
											glUniform3f(MaterialDiffuse,
													obj.mas[i].Kd[0],
													obj.mas[i].Kd[1],
													obj.mas[i].Kd[2]);
										}

										// glUniform3f(MaterialAmbient, 0.1f, 0.1f, 0.1f);
										glUniform3f(MaterialAmbient,
												obj.mas[i].Ka[0],
												obj.mas[i].Ka[1],
												obj.mas[i].Ka[2]);

										glUniform3f(MaterialSpecular,
												obj.mas[i].Ks[0],
												obj.mas[i].Ks[1],
												obj.mas[i].Ks[2]);

										if (obj.meshes[i].istex) {
											// Bind our texture in Texture Unit 0
											glActiveTexture(
											GL_TEXTURE0 + tid + i);
											glBindTexture(GL_TEXTURE_2D,
													textures[i]);

											// Set our "myTextureSampler" sampler to use Texture Unit 0
											glUniform1i(TextureID, tid + i);
											glUniform1f(istex, 1.0f);
										} else {
											glUniform1f(istex, -1.0f);
										}

										// Draw the triangles !
										glDrawArrays(GL_TRIANGLES, 0,
												obj.meshes[i].vertices.size());
									}

									// Bind our texture in Texture Unit 1
									glActiveTexture(GL_TEXTURE0 + tid - 1);
									glBindTexture(GL_TEXTURE_2D, tex_saver);
									glCopyTexSubImage2D(GL_TEXTURE_2D, 0,
											xshift, yshift, 0, 0, width,
											height);

									// saveim(tmpdata, tmpshow, tmpsave, height,
									//	width, folder, "single", renderid);
								}
							}

							//////////////////////////////////
							// do we need this?
							glBindTexture(GL_TEXTURE_2D, 0);

							// map buffer objects to get CUDA device pointers
							CUDA_CHECK_RETURN(
									cudaGraphicsMapResources(1,
											&cuda_tex_saver_resource, 0));
							//printf("Mapping tex_in\n");

							// texture pointer
							cudaArray *in_array;
							CUDA_CHECK_RETURN(
									cudaGraphicsSubResourceGetMappedArray(
											&in_array, cuda_tex_saver_resource,
											0, 0));
							// cout << static_cast<int>(in_array)<<endl;

							/////////////////////////////////////////////////////////////////////////////
							// execute CUDA kernel
							// cudaArray *g_data_array,	float *g_output_data, int timebin, int imgh, int imgw, int sz
							launch_cudaProcess2(in_array,
									cuda_dest_saver_resource, timebin, height,
									width, maxsz, 100 * TEN, 100 * TBE);
							////////////////////////////////////////////////////////////////////////////

							CUDA_CHECK_RETURN(
									cudaGraphicsUnmapResources(1,
											&cuda_tex_saver_resource, 0));
							// all the block
						}
					}

					auto end = std::chrono::high_resolution_clock::now();

					std::chrono::duration<double, std::milli> fp_ms = end
							- start;

					cout << "rendering time (ms)" << fp_ms.count() << endl;

					cudaMemcpy(result, cuda_dest_saver_resource, size_tex_data,
							cudaMemcpyDeviceToHost);

					savedistre(result, maxval, maxdst, mindst, tmpsaveall,
							height, width, timebin, folder, "light", conf);

					// all the light
				}
			}

			// confocal or not
		}

		// all the rotation
	}
}

