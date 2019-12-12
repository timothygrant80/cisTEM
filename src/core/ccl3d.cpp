//Connected-component labeling
#include "core_headers.h"

ccl3d::ccl3d(Image &input3d)
{
	parent = new int[input3d.logical_x_dimension * input3d.logical_y_dimension * input3d.logical_z_dimension]();
}

ccl3d::~ccl3d()
{
    delete [] parent;
}

int ccl3d::Max(int i,int j,int k)
{
	if(i >= j && i >= k) return i;
	else if(j >= i && j >= k) return j;
	else return k;
}
int ccl3d::Med(int i,int j,int k)
{
	if(i >= j && i <= k || i <= j && i >= k) return i;
	else if(j >= i && j <= k || j <= i && j >= k) return j;
	else return k;
}
int ccl3d::Min(int i,int j,int k)
{
	if(i <= j && i <= k) return i;
	else if(j <= i && j <= k) return j;
	else return k;
}

int ccl3d::Find(int x, int parent[])
{
	int k = x;
	while (0 != parent[k])
		k = parent[k];
	int i=x , j;
	while( i != k )
	{
		j = parent[ i ];
		parent[ i ]= k;
		i=j;
	}
	return k;
}

float mostFrequent(wxVector<float> arr, int n)
{
	// Sort the array
	wxVectorSort(arr);
	// find the max frequency using linear traversal
	int max_count = 1, curr_count = 1;
	float res = arr[0];
	for (int i = 1; i < n; i++)
	{
		if (arr[i] == arr[i - 1])
			curr_count++;
		else
		{
			if (curr_count > max_count)
			{
				max_count = curr_count;
				res = (float)arr[i - 1];
			}
			curr_count = 1;
		}
	}
	// If last element is most frequent 
	if (curr_count > max_count) 
	{ 
		max_count = curr_count; 
		res = arr[n - 1]; 
	} 
  
	return res; 
} 

void ccl3d::Union(int big, int small, int parent[])
{
	int i = big;
	int j = small;
	while (0 != parent[i])
		i = parent[i];
	while (0 != parent[j])
		j = parent[j];
	if (i != j)
		parent[i] = j;
}

void ccl3d::Two_noZero(int a, int b, int c)
{
	if (a == 0)
	{
		if(b < c) Union(c, b, parent);
		if(c < b) Union(b, c, parent);
	}
}
void ccl3d::GetLargestConnectedDensityMask(Image &input3d, Image &output_largest_connected_density3d)
{
	//first pass
	int label =0;
	long pixel_counter = 0;
	long pixel_new = 0;
	float max;
	output_largest_connected_density3d.Allocate(input3d.logical_x_dimension, input3d.logical_y_dimension, input3d.logical_z_dimension, true);

	for (int z = 0; z < input3d.logical_z_dimension; z++)
	{
		for (int y = 0; y < input3d.logical_y_dimension; y++)
		{
			for (int x = 0; x < input3d.logical_x_dimension; x++)
			{
				output_largest_connected_density3d.real_values[pixel_new] = 0.0;
				pixel_new++;
			}
			pixel_new += input3d.padding_jump_value;
		}
	}
	for (int z =0; z < input3d.logical_z_dimension; z++)
	{
		for (int y =0; y < input3d.logical_y_dimension; y++)
		{
			for (int x = 0; x < input3d.logical_x_dimension; x++)
			{
				if (input3d.real_values[pixel_counter] != 0)
				{
					int left = x - 1 < 0? 0 : output_largest_connected_density3d.real_values[output_largest_connected_density3d.ReturnReal1DAddressFromPhysicalCoord(x - 1, y, z)];
					int up = y - 1 < 0? 0 : output_largest_connected_density3d.real_values[output_largest_connected_density3d.ReturnReal1DAddressFromPhysicalCoord(x, y - 1, z)];
					int ahead  = z - 1 < 0? 0 : output_largest_connected_density3d.real_values[output_largest_connected_density3d.ReturnReal1DAddressFromPhysicalCoord(x, y, z - 1)];
					if (left != 0 || up != 0 || ahead != 0)
					{
						if (left * up == 0 && left * ahead == 0 && up * ahead == 0) output_largest_connected_density3d.real_values[pixel_counter] = Max(left, up, ahead);// Only one none-zero
						else
						{
							if (left * up != 0 && left * ahead != 0 && up * ahead != 0) // None zero
							{
								int min3 = Min(left, up, ahead);
								output_largest_connected_density3d.real_values[pixel_counter] = min3;
								if ( min3 < left) Union(left, min3, parent);
								if ( min3 < up) Union(up, min3, parent);
								if ( min3 < ahead) Union(ahead, min3, parent);
							}
							else // One zero
							{
								int med3 = Med(left, up, ahead);
								output_largest_connected_density3d.real_values[pixel_counter] = med3;
								Two_noZero(left, up, ahead);
								Two_noZero(ahead, left, up);
								Two_noZero(up, ahead, left);
							}
						}
					}
					else output_largest_connected_density3d.real_values[pixel_counter] = ++ label;
				}
				pixel_counter++;
			}
			pixel_counter += input3d.padding_jump_value;
		}
	}
	//second pass
	wxVector<float> pixel_value;
	long pixel_counter1 = 0;
	for (int z = 0; z < input3d.logical_z_dimension; z++)
	{
		for (int y = 0; y < input3d.logical_y_dimension; y++)
		{
			for (int x = 0; x < input3d.logical_x_dimension; x++)
			{
				if (input3d.real_values[pixel_counter1] == 1)
				{
					output_largest_connected_density3d.real_values[pixel_counter1] = Find(output_largest_connected_density3d.real_values[pixel_counter1], parent);
					pixel_value.push_back(output_largest_connected_density3d.real_values[pixel_counter1]);
				}
				pixel_counter1++;
			}
			pixel_counter1 += input3d.padding_jump_value;
		}
	}
	max = mostFrequent(pixel_value, pixel_value.size());
//	wxPrintf("%f\n", max);

	long pixel_counter2 = 0;
	for (int z = 0; z < input3d.logical_z_dimension; z++)
	{
		for (int y = 0; y < input3d.logical_y_dimension; y++)
		{
			for (int x = 0; x < input3d.logical_x_dimension; x++)
			{
				if (output_largest_connected_density3d.real_values[pixel_counter2] == max) output_largest_connected_density3d.real_values[pixel_counter2] = 1.0f;
				else output_largest_connected_density3d.real_values[pixel_counter2] = 0.0f;
				pixel_counter2++;
			}
			pixel_counter2 += input3d.padding_jump_value;
		}
	}
}
