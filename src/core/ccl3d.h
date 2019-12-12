
class ccl3d {

public:

	int* parent;
	ccl3d(Image &input3d);
	~ccl3d();
	int Max(int i,int j,int k);
	int Med(int i,int j,int k);
	int Min(int i,int j,int k);
	int Find(int x, int parent[]);
	void Union(int big, int small, int parent[]);
	void Two_noZero(int a, int b, int c);
	void GetLargestConnectedDensityMask(Image &input3d, Image &output_largest_connected_density3d);
};

