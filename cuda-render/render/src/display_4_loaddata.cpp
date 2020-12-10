#include "renderclass.h"

#include <map>
#include <fstream>
#include <assert.h>

//////////////////////////////////////////////////////
// this is used to get meaningful data
// input "f     1/2/3  1/2/3 1/2/3    "
// output "f" "1/2/3" "1/2/3" "1/2/3"
void SplitString(const std::string &s, std::vector<std::string> &v,
		const std::string &c) {
	std::string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = s.find(c);
	while (std::string::npos != pos2) {
		if (pos1 < pos2)
			v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}

	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

// this is used to get all data
// input "1//"
// output "1" "" ""
void SplitString2(const std::string &s, std::vector<std::string> &v,
		const std::string &c) {
	std::string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = s.find(c);
	vector<int> pos;
	while (std::string::npos != pos2) {
		pos.push_back(pos2);
		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}

	if (pos.size() == 0) {
		v.push_back(s);
		return;
	}

	int be = -1;
	int en = -1;
	for (int i = 0; i < pos.size(); i++) {
		if (i == 0) {
			be = 0;
			en = pos[i];
		} else {
			be = pos[i - 1] + c.size();
			en = pos[i];
		}
		v.push_back(s.substr(be, en - be));
	}
	v.push_back(s.substr(pos[pos.size() - 1] + c.size()));
}

////////////////////////////////////////////////////////////////////
//mtl
bool readmtl(string folder, string mtlname, vector<material> &mas,
		vector<cv::Mat> &ims, map<string, int> &m_map //matMap[material_name] = material_ID
		) {
	string fullname = folder + "/" + mtlname;
	ifstream fp(fullname.c_str());
	if (fp.fail()) {
		cout << "error, cannot load mtl file!" << endl;
		cout << fullname << endl;
		// system("pause");
		return false;
	}

	// mas has no material
	assert(mas.size() == 0);
	assert(ims.size() == 0);
	assert(m_map.size() == 0);

	vector<material> all_list;
	int midx = -1;
	int mconponent = 5;

	string line;
	while (getline(fp, line)) {
		vector<string> parts;
		SplitString(line, parts, " ");
		if (parts.size() < 2)
			continue;
		if (parts[0] == "#")
			continue;

		if (parts[0] == "newmtl") {
			assert(parts.size() == 2);
			// assert(mconponent == 5);

			material tmp;
			tmp.mtlname = parts[1];
			all_list.push_back(tmp);
			midx += 1;
			mconponent = 0;
		}

		if (parts[0] == "Ka") {
			assert(parts.size() == 4);
			for (int i = 0; i < 3; i++) {
				double kai = stod(parts[1 + i]);
				// assert(kai == 0);
				/*
				 if (kai != 0) {
				 cout << "non zero ambient" << endl;
				 return false;
				 }
				 */
				all_list[midx].Ka[i] = kai;
			}
			mconponent++;
		}

		if (parts[0] == "Kd") {
			assert(parts.size() == 4);
			for (int i = 0; i < 3; i++) {
				double kdi = stod(parts[1 + i]);
				assert(kdi >= 0);
				all_list[midx].Kd[i] = kdi;
			}
			mconponent++;
		}

		if (parts[0] == "Ks") {
			assert(parts.size() == 4);
			for (int i = 0; i < 3; i++) {
				double ksi = stod(parts[1 + i]);
				assert(ksi >= 0);
				all_list[midx].Ks[i] = ksi;
			}
			mconponent++;
		}

		if (parts[0] == "Ns") {
			assert(parts.size() == 2);
			int ns = stoi(parts[1]);
			// assert(ns == 10);
			/*
			 if (ns != 10) {
			 cout << "ns is not 10" << endl;
			 return false;
			 }
			 */
			all_list[midx].Ns = ns;
			mconponent++;
		}

		if (parts[0] == "illum") {
			assert(parts.size() == 2);
			int illum = stoi(parts[1]);
			// assert(illum == 2);
			all_list[midx].illum = illum;
			mconponent++;
		}

		if (parts[0] == "map_Kd") {
			assert(parts.size() == 2);
			string filename = folder + "/" + parts[1];
			all_list[midx].map_Kd = filename;
			all_list[midx].istex = true;
		}
	}
	fp.close();

	// sort mas!
	vector<material> gray_list;
	vector<material> tex_list;
	for (int i = 0; i < all_list.size(); i++)
		if (all_list[i].istex) {
			// make sure that it is valid texutre
			cv::Mat im = cv::imread(all_list[i].map_Kd);
			// assert(!im.empty());
			if (im.empty()) {
				cout << "empty image" << endl;
				return false;
			}

			if ((im.type() != CV_8UC1) && (im.type() != CV_8UC3)) {
				cout << "wrong type image" << endl;
				return false;
			}

			tex_list.push_back(all_list[i]);
			ims.push_back(im);

		} else
			gray_list.push_back(all_list[i]);

	for (int i = 0; i < tex_list.size(); i++)
		mas.push_back(tex_list[i]);
	for (int i = 0; i < gray_list.size(); i++)
		mas.push_back(gray_list[i]);

	for (int i = 0; i < mas.size(); i++)
		m_map[mas[i].mtlname] = i;

	return true;
}

bool readobj(string folder, string obj_name, vector<glm::vec3> &v_list,
		vector<glm::vec2> &vt_list, vector<glm::i32vec3> &fv_list,
		vector<glm::i32vec3> &fvt_list, vector<int> &fm_list,
		vector<material> &m_list, vector<cv::Mat> &ims) {

	string fullname = folder + "/" + obj_name;
	ifstream fp(fullname.c_str());
	if (fp.fail()) {
		cout << "error, cannot load obj file!" << endl;
		// system("pause");
		return false;
	}

	// ���������������������������������������������
	// mtl, use, v, vt, f
	float xmin = 10000, ymin = 10000, zmin = 10000, xmax = -10000,
			ymax = -10000, zmax = -10000;

	map<string, int> material_map;	// matMap[material_name] = material_ID
	int current_material = -1;

	string line;
	while (getline(fp, line)) {
		vector<string> parts;
		SplitString(line, parts, " ");
		if (parts.size() < 2)
			continue;
		if (parts[0] == "#")
			continue;

		if (parts[0] == "mtllib") {
			assert(parts.size() == 2);
			bool suc = readmtl(folder, parts[1], m_list, ims, material_map);
			if (!suc) {
				cout << "error, cannot load mtl file!" << endl;
				// system("pause");
				return false;
			}
		}

		if (parts[0] == "v") {
			assert(parts.size() == 4);
			// cout<<"v\t"<<parts[1]<<"\t"<<parts[2]<<"\t"<<parts[3]<<"\t"<<endl;
			glm::vec3 tmp(stod(parts[1]), stod(parts[2]), stod(parts[3]));
			/*
			 tmp.x = tmp.x * 5;
			 tmp.y = (tmp.y - 0.12) * 5;
			 tmp.z = tmp.z * 5;
			 */
			if (tmp[0] < xmin)
				xmin = tmp[0];
			if (tmp[0] > xmax)
				xmax = tmp[0];
			if (tmp[1] < ymin)
				ymin = tmp[1];
			if (tmp[1] > ymax)
				ymax = tmp[1];
			if (tmp[2] < zmin)
				zmin = tmp[2];
			if (tmp[2] > zmax)
				zmax = tmp[2];
			v_list.push_back(tmp);
		}

		if (parts[0] == "vt") {
			assert(parts.size() == 3 || parts.size() == 4);
			if (parts.size() == 4)
				assert(stod(parts[3]) == 0.0f);
			// cout<<"vt\t"<<parts[1]<<"\t"<<parts[2]<<"\t"<<parts[3]<<"\t"<<endl;
			glm::vec2 tmp(stod(parts[1]), stod(parts[2]));
			vt_list.push_back(tmp);
		}

		if (parts[0] == "usemtl") {
			// what happens if the key does not
			// exist in the map?
			assert(parts.size() == 2);
			map<string, int>::iterator iter;
			iter = material_map.find(parts[1]);
			assert(iter != material_map.end());
			current_material = iter->second;
		}

		if (parts[0] == "f") {
			// triangles!
			assert(parts.size() == 4);
			glm::ivec3 fv(0, 0, 0);
			glm::ivec3 fvt(0, 0, 0);
			bool faceparse = true;

			for (int i = 0; i < 3; i++) {
				vector<string> nums;
				SplitString2(parts[i + 1], nums, "/");

				// v/vt/n
				fv[i] = stoi(nums[0]);

				if (m_list[current_material].istex) {
					// if (debug)
					// assert(nums[1] != "");
					// it should be texture cood but there is no coord
					// continue
					if (nums[1] == "") {
						faceparse = false;
						break;
					}
					fvt[i] = stoi(nums[1]);
				}
			}

			if (faceparse && fv[0] != fv[1] && fv[0] != fv[2]
					&& fv[1] != fv[2]) {
				fm_list.push_back(current_material);
				fv_list.push_back(fv);
				fvt_list.push_back(fvt);
			}
		}
	}
	fp.close();

	if (xmin < -0.5 || xmax > 0.5 || ymin < -0.5 || ymax > 0.5 || zmin < -0.5
			|| zmax > 0.5)
		return false;

	// resize
	cout << "xmin\t" << xmin << "\txmax" << xmax << endl;
	cout << "ymin\t" << ymin << "\tymax" << ymax << endl;
	cout << "zmin\t" << zmin << "\tzmax" << zmax << endl;

	glm::vec3 shift = glm::vec3((xmin + xmax) / 2, (ymin + ymax) / 2,
			(zmin + zmax) / 2);
	float scale = std::max(zmax - zmin, std::max(ymax - ymin, xmax - xmin));
	/*
	 scale = 2
	 * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.2
	 + 0.5) / scale;
	 */
	float scale2 = 0.6
			+ static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.6;

	xmin = 10000;
	ymin = 10000;
	zmin = 10000;
	xmax = -10000;
	ymax = -10000;
	zmax = -10000;
	for (int i = 0; i < v_list.size(); i++) {
		glm::vec3 tmp = (v_list[i] - shift) * scale2 / scale;
		v_list[i] = tmp;

		if (tmp[0] < xmin)
			xmin = tmp[0];
		if (tmp[0] > xmax)
			xmax = tmp[0];
		if (tmp[1] < ymin)
			ymin = tmp[1];
		if (tmp[1] > ymax)
			ymax = tmp[1];
		if (tmp[2] < zmin)
			zmin = tmp[2];
		if (tmp[2] > zmax)
			zmax = tmp[2];
	}

	// resize
	cout << "xmin\t" << xmin << "\txmax" << xmax << endl;
	cout << "ymin\t" << ymin << "\tymax" << ymax << endl;
	cout << "zmin\t" << zmin << "\tzmax" << zmax << endl;
	cout << "scale\t" << scale << endl;

	return true;
}

// if use texture
// then all the faces must have texture index
// even if the face may use mtl that do not have any texture!!!
mesh render::loadobj(string folder, string obj_name, bool &suc) {
	vector<glm::vec3> v_list;
	vector<glm::vec2> vt_list;
	vector<glm::i32vec3> fv_list;
	vector<glm::i32vec3> fvt_list;
	vector<int> fm_list;
	vector<material> m_list;
	vector<cv::Mat> ims;

	bool objsuc = readobj(folder, obj_name, v_list, vt_list, fv_list, fvt_list,
			fm_list, m_list, ims);
	if (!objsuc) {
		suc = false;
		cout << "unsuccessful loading obj!" << endl;
		mesh tmp;
		return tmp;
	}

	// calculate mesh
	vector<glm::vec3> pointnormals;
	pointnormals.resize(v_list.size());
	for (int i = 0; i < v_list.size(); i++)
		pointnormals[i] = glm::vec3(0, 0, 0);

	vector<glm::vec3> facenormals;
	facenormals.resize(fv_list.size());
	for (int i = 0; i < fv_list.size(); i++) {
		glm::vec3 p0 = v_list[fv_list[i][0] - 1];
		glm::vec3 p1 = v_list[fv_list[i][1] - 1];
		glm::vec3 p2 = v_list[fv_list[i][2] - 1];

		glm::vec3 v01 = p1 - p0;
		glm::vec3 v02 = p2 - p0;
		facenormals[i] = glm::cross(v01, v02);

		pointnormals[fv_list[i][0] - 1] += facenormals[i];
		pointnormals[fv_list[i][1] - 1] += facenormals[i];
		pointnormals[fv_list[i][2] - 1] += facenormals[i];
	}

	// how many material in total
	int texnum = 0;
	vector<group> parts;
	int fsize = fm_list.size();
	for (int i = 0; i < m_list.size(); i++) {
		group tmp;
		if (m_list[i].map_Kd != "") {
			texnum++;
			tmp.istex = true;
			tmp.texid = i;
		} else {
			tmp.istex = false;
			tmp.texid = -1;
		}
		parts.push_back(tmp);
	}

	int currentp = -1;
	for (int i = 0; i < fm_list.size(); i++) {
		currentp = fm_list[i];

		// Get the indices of its attributes
		int vertexIndex0 = fv_list[i][0];
		int vertexIndex1 = fv_list[i][1];
		int vertexIndex2 = fv_list[i][2];

		glm::vec3 vertex0 = v_list[vertexIndex0 - 1];
		glm::vec3 vertex1 = v_list[vertexIndex1 - 1];
		glm::vec3 vertex2 = v_list[vertexIndex2 - 1];

		// Put the attributes in buffers
		parts[currentp].vertices.push_back(vertex0);
		parts[currentp].vertices.push_back(vertex1);
		parts[currentp].vertices.push_back(vertex2);

		glm::vec3 diff1 = vertex1 - vertex0;
		glm::vec3 diff2 = vertex2 - vertex0;
		glm::vec3 cross = glm::cross(diff1, diff2);
		glm::vec3 normal = glm::normalize(cross);

		// bunny we use smooth nriomal
		/*
		parts[currentp].normals.push_back(normal);
		parts[currentp].normals.push_back(normal);
		parts[currentp].normals.push_back(normal);
      */

		 parts[currentp].normals.push_back(pointnormals[vertexIndex0 - 1]);
		 parts[currentp].normals.push_back(pointnormals[vertexIndex1 - 1]);
		 parts[currentp].normals.push_back(pointnormals[vertexIndex2 - 1]);


		if (parts[currentp].istex) {
			for (int k = 0; k < 3; k++) {
				int uvIndex = fvt_list[i][k];
				assert(uvIndex > 0);
				glm::vec2 uv = vt_list[uvIndex - 1];
				parts[currentp].uvs.push_back(uv);
			}
		}
	}

	for (int i = 0; i < parts.size(); i++)
		cout << i << "\t" << parts[i].vertices.size() << endl;

	mesh re;
	re.vertices = v_list;
	re.ims = ims;
	re.meshes = parts;
	re.mas = m_list;
	re.texnum = texnum;
	return re;
}

