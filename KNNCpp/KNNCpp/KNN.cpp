//KNN algorithm for titanic.dat

//头文件 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int Random(int x) {
	return rand() % x;
}
//Data stucture
struct human
{
	vector<double> attribute;
	double label;
	double distance = -1;
};

template<typename T>
bool notin(T index, vector<T>indexlist) {
	for (int i = 0; i < indexlist.size(); i++)
	{
		if (index == indexlist[i]) return false;
	}
	return true;
}

double distance(vector<double> a, vector<double>b) {
	if (a.size() != b.size()) {
		cout << "The dimensions of the vector are not the same" << endl;
		return -1;
	}
	int n = a.size();
	double distance = 0;
	for (int i = 0; i < n; i++)
	{
		distance += (a[i] - b[i])*(a[i] - b[i]);
	}
	return sqrt(distance);
}

bool LessSort(human a, human b)
{
	return (a.distance < b.distance);
}


//1. Read the titanic.dat file
int readDat(const char* file, vector<human> &people) {
	string buf,temp;
	ifstream fd(file, ios::in);
	if (!fd)
	{
		cout << "Error in Opening " << file << endl;
		return 1;
	}
	int n = 0;
	while (fd.good())
	{
		n++;
		buf.clear();
		getline(fd, buf);
		human person;
		//read .dat file
		if (buf[0] == '@' || buf.size() < 2) continue;
		stringstream input(buf);
		while (getline(input, temp, ',')) {
			double value;
			istringstream iss(temp);
			iss >> value;
			person.attribute.push_back(value);
		}
		//cout << person.attribute.size() << endl;
		if (person.attribute.size() > 3) {
			person.label = person.attribute[3];
			person.attribute.pop_back();
		}
		else { 
			cout << "Attributes are missing" << endl;
			return 1;
		}
		
		//cout << person.attribute[0] << " " << person.attribute[1] << " " << person.attribute[2] << " " << person.label << endl;
		//cout << temp << endl;
		people.push_back(person);
	}
	return 0;
}

//2. Standardlise the dataset, using max-minimum standardlisation, to [0,1]
int DataStandard(vector<human> &people) {
	vector<double> standClass,standAge,standSex;
	double maxclass = -DBL_MAX; double maxage = -DBL_MAX; double maxsex = -DBL_MAX;
	double minclass = DBL_MAX; double minage = DBL_MAX; double minsex = DBL_MAX;
	for (int i = 0; i < people.size(); i++)
	{
		if (people[i].attribute[0] > maxclass) maxclass = people[i].attribute[0];
		if (people[i].attribute[0] < minclass) minclass = people[i].attribute[0];
		if (people[i].attribute[1] > maxage) maxage = people[i].attribute[1];
		if (people[i].attribute[1] < minage) minage = people[i].attribute[1];
		if (people[i].attribute[2] > maxsex) maxsex = people[i].attribute[2];
		if (people[i].attribute[2] < minsex) minsex = people[i].attribute[2];
	}
	//cout << (".2f",maxclass) << " " << (".2f",maxage) << " " << (".2f",maxsex) << endl;
	//cout << (".2f",minclass) << " " << (".2f", minage) << " " << (".2f", minsex) << endl;
	for (int j = 0; j < people.size(); j++) {
		people[j].attribute[0] = (people[j].attribute[0] - minclass) / (maxclass - minclass);
		people[j].attribute[1] = (people[j].attribute[1] - minage) / (maxage - minage);
		people[j].attribute[2] = (people[j].attribute[2] - minsex) / (maxsex - minsex);
		if (people[j].attribute[0] > 1 || people[j].attribute[0] < 0 || people[j].attribute[1] > 1 || people[j].attribute[1] < 0 ||
			people[j].attribute[2] > 1 || people[j].attribute[2] < 0) {
			cout << "Standardlised value is out of range [0,1]" << endl;
			return 1;
		}
	}
	return 0;
}

//3. Data are divided into two sets, traindata and testdata
void DataDivide(vector<human> &people, vector<human> &traindata, vector<human> &testdata, double ratio) {
	vector<int> indexlist;
	for (int i = 0; i < round(people.size()*ratio); i++) {
		int index = Random(people.size());
		while (index < people.size() && index >= 0) {
			if ( notin(index, indexlist) ) {
				indexlist.push_back(index);
				break;
			}
			else index = Random(people.size());
		}
	}
	for (int i = 0; i < people.size(); i++)
	{
		if (notin(i, indexlist)) testdata.push_back(people[i]);
		else traindata.push_back(people[i]);
	}

}

//4. KNN classification starts
//output: the label
double KNNClassify(vector<human> traindata, human sample, int k) {
	vector<double> labellist;
	vector<int> labelnum;
	int index;
	double maxlabel;
	for (int i = 0; i < traindata.size(); i++)
	{
		double dis = distance(traindata[i].attribute, sample.attribute);
		traindata[i].distance = dis;
	}
	sort(traindata.begin(), traindata.end(), LessSort);    //进行升序排序
	//analyse the ratio of the label
	labellist.push_back(traindata[0].label);
	labelnum.push_back(1);
	for (int j = 1; j < k; j++){
		int judge = 0;
		for (int i = 0; i < labellist.size(); i++) {
			//cout << traindata[j].label << " " << labellist[i] << endl;
			if (traindata[j].label == labellist[i]) {
				labelnum[i]++; 
				judge = 1; 
				break;
			}
		}
		if (judge == 1) continue;
		else {
			labellist.push_back(traindata[j].label);
			labelnum.push_back(1);
		}
	}
	//report the labellist destribution
	//cout << labellist[0] << " " << labelnum[0] << " | " << labellist[1] << " " << labelnum[1] << endl;

	//find the most frequent label and return
	index = NULL;
	maxlabel = 0;
	for (int i = 0; i < labellist.size(); i++)
	{
		if (labelnum[i] > maxlabel){
			maxlabel = labelnum[i];
			index = i;
		}
	}
	//cout << labellist[index] << endl;;
	return labellist[index];
}

//5. KNN classification test using testdata set
//output: the hit rate of the classification (the true ratio)
double KNNClassTest(vector<human> traindata, vector<human> testdata,int k) {
	double n = testdata.size();
	double truenum = 0;
	//vector<double> ClassResult;
	for (int i = 0; i < n; i++)
	{
		double classify = KNNClassify(traindata, testdata[i], k);
		//ClassResult.push_back(classify);
		//cout << classify << " " << testdata[i].label << endl;
		if (classify == testdata[i].label) truenum++;
	}
	return (truenum/n);
}

int main(int argc, char* argv) {
	vector<human> people;
	vector<human> traindata;
	vector<human> testdata;
	int error = 0;
	double ratio;

	if (argc == 1) {
		cout << "KNN.py: A KNN algorithm program to predict the survival of a person in the titanic incident.\n" << endl;
		cout << "USage:\nKNN.py filepath k [DataDivideRatio=0.7]\n" << endl;
		cout << "filepath\n\tthe full path of the .dat or .csv file, relative or absolute all both accepted." << endl;
		cout << "k\n\tthe number of nearest neighbor, which has to be set by the user." << endl;
	}
	else if (argc == 2) {
		if ((int)(argv[1]) >= 48 && (int)(argv[1]) <= 57){
			cout << "Please specify the filepath." << endl;
		}
		else {
			cout << "Please specify the k number. Less than 50 is recommended." << endl;
		}	
	}
	else if (argc > 2) {
		int k = int(argv[2]);
		if (argc == 3) {
			ratio = 0.7;
		}
		else {
			ratio = (double)argv[3];
		}
		//KNN procure starts
		error = readDat("titanic.dat", people);
		if (error == 1) {
			cout << "Read data failed." << endl;
		}
		error = DataStandard(people);
		if (error == 1) {
			cout << "Standardlisation failed." << endl;
		}
		DataDivide(people, traindata, testdata, 0.7);
		
		//---Test division result--//
		//cout << traindata.size() << " " << testdata.size() << endl;

		//Calculate the true prediction rate of KNN classification
		double rate = KNNClassTest(traindata, testdata, 10);
		cout << rate << endl;

		//---Test classification result--//
		//double label = KNNClassify(traindata, testdata[0], 10);
		//cout << label << " " << testdata[0].label << endl;
	}
	system("pause");
	return 0;
}