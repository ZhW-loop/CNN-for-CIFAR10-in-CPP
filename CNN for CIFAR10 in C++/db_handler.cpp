#include "db_handler.h"
//  10000¸ö  50000¸ö
vector<Array3d> Data_Handler::read_CIFAR10_image(string filename) const
{	
	fstream f(filename);
	vector<double> v;
	vector<Array3d> ret;
	double v_tmp = 0.0;
	int step = 0;
	while (!f.eof())
	{	
		v.clear();
		for (int i = 0; i < 32 * 32 * 3; ++i)
		{
			f >> v_tmp;
			v.push_back(v_tmp);
		}
		Array3d temp(32, 32, 3, v);
		ret.push_back(temp);
		step++;
		if (step == 10) break;
	}
	f.close();
	return ret;
}

vector<int> Data_Handler::read_CIFAR10_label(string filename) const
{
	fstream f(filename);
	vector<int > ret;
	int ret_tmp = 0;
	int step = 0;
	while (!f.eof())
	{
		f >> ret_tmp;
		ret.push_back(ret_tmp);
		step++;
		if (step == 10) break;
	}
	return ret;
}
/*
int main()
{
	Data_Handler m;
	vector<int> test = m.read_CIFAR10_label("./CIFAR10_for_C++_train_label.txt");
	cout << test.size();
}
*/