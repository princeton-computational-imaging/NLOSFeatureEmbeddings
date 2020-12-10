#include <iostream>
using namespace std;

#include <vector>
#include <string>
#include <fstream>

#include "renderclass.h"

void getFiles(string parentFolder, vector<string> &vecFileNames) {

	string cmd = "ls " + parentFolder + " > temp.log";
	system(cmd.c_str());

	ifstream ifs("temp.log");
	//������������������log���������������������������������������������������������������
	if (ifs.fail()) {
		return;
	}

	string fileName;
	while (getline(ifs, fileName)) {
		vecFileNames.push_back(parentFolder + "/" + fileName);
	}

	ifs.close();
	return;
}

int main() {

	string parentFlder = "../../data/bunny-model";

	string parentSvFolder = "../../data/bunny-renders";

	// to change to shapenet models
	// parentFlder = "/u6/a/wenzheng/remote2/datasets/shapenet/ShapeNetCore.v2/03790512";
	// parentSvFolder = "/u6/a/wenzheng/remote3/dataset-generated/bikenonfocal-shape_0.6_1.2-shift_0.4_0.4_-0.4_0.4";

	vector<string> folders;
	getFiles(parentFlder, folders);

	// use predefined rotation and shift or not
	// in this case, we do not rotate the model
	bool definerot = true;
	float rotx = 0;
	float roty = 0;
	float rotz = -0;
	float shiftx = 0;
	float shifty = 0;
	float shiftz = -0;

	/*
	 * To render a model with specific rotation parameters
	 string mdname = "2d655fc4ecb2df2a747c19778aa6cc0";
	 folders.clear();
	 folders.push_back(parentFlder + "/" + mdname);
	 float rotx = 2.9725;
	 float roty = -88.7643;
	 float rotz = -7.4680;
	 float shiftx = 0.1451;
	 float shifty = -0.1690;
	 float shiftz = -0.0214;
	 */

	char cmd[256];
	sprintf(cmd, "mkdir %s", parentSvFolder.c_str());
	system(cmd);

	for (int i = 0; i < 1; i++) {
		string svfolder = parentSvFolder + "/" + to_string(i);

		char cmd[256];
		sprintf(cmd, "mkdir %s", svfolder.c_str());
		system(cmd);
	}

	int height = 256;
	int width = 256;

	// each model, we sample 100x100, the more the better, but takes longer time.
	int samplenum = 600;
	// we sample directions over a semisphjere, however, many direction is meaningful, so we can discard some directions
	// 0.8 means we sample based on a cone instead of a semisphere.
	float sampleratio = 0.8;

	// we will render each sample individually and copy it to a big texture. Finally we merge it.
	// the big texture map size is maxsz * height.
	int blocksz = 40;

	render *tmp = new render(height, width, blocksz, sampleratio);
	tmp->initializecuda();
	tmp->programobj();

	// how many renders do we want
	// in this example, we render only one model.
	int rendernum = 1;
	int i = -1;
	int step = 0;

	// for (int i = 0; i < folders.size(); i++) {
	while (true) {

		i = (i + 1) % folders.size();

		string folder = folders[i] + "/model";
		string name = "model_normalized.obj";

		// string folder = folders[i];
		//	string name = "model.obj";

		bool suc = true;
		mesh tmpobj = tmp->loadobj(folder, name, suc);
		if (!suc) {
			continue;
		} else {

			tmp->loadmesh(tmpobj);

			// how many rotation we have
			int rnum = 1;
			// how many lights we set, only 1 light for non confocal case.
			int lvnum = 1;
			int lhnum = 1;

			int shininesslevel = 0;
			for (shininesslevel = 0; shininesslevel < 1; shininesslevel++) {

				int pos = folders[i].find_last_of('/');
				string svfolder = parentSvFolder + "/"
						+ to_string(shininesslevel) + "/"
						+ folders[i].substr(pos + 1,
								folders[i].length() - pos - 1);
				cout << folder << "\t" << name << endl;
				cout << "svfolder\t" << svfolder << endl;

				char cmd[256];
				sprintf(cmd, "mkdir %s", svfolder.c_str());
				system(cmd);

				// say the smapling number is 80x80
				// the blacksz is 40
				// so we need 2x2 block
				// hnum and vnum should be divided by blocksz!!!
				int actualsample = samplenum * sampleratio;
				int sz = blocksz;
				assert(actualsample % sz == 0);
				int hnum = actualsample;
				int vnum = actualsample;

				tmp->display(svfolder, tmpobj, shininesslevel, sz, rnum, lhnum,
						lvnum, hnum, vnum, definerot, rotx, roty, rotz, shiftx,
						shifty, shiftz);
			}

			tmp->deletemesh();

			step++;
			if (step >= rendernum)
				break;
		}
	}

	delete tmp;
	cout << "done!" << endl;
	return 0;
}

