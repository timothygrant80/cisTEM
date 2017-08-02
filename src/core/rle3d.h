// class for run length incoding - copied and modified from TIGRIS

class rle3d_coord {

public:

	int x_pos;
	int y_pos;
	int z_pos;
	long length;
	int group_number; // this is only used in certain operations, e.g. when working out connected rles.

	rle3d_coord();
};

class rle3d {

private:

	long allocated_coordinates;
	long number_of_coordinates;

public :

	long x_size;
	long y_size;
	long z_size;

	long number_of_groups;

	rle3d_coord *rle_coordinates;

	rle3d();
	~rle3d();
	rle3d(Image &input3d);

	void AddCoord(long x, long y, long z, long current_length);

	void EncodeFrom(Image &input3d);
	//void DecodeTo(Image &output3d);
	void ConnectedSizeDecodeTo(Image &output3d);
	void Write(const char *filename);
	void GroupConnected();

	inline void SameGroup(long first, long second)
	{
		if (rle_coordinates[first].group_number == 0 && rle_coordinates[second].group_number != 0)
		{
			rle_coordinates[first].group_number = rle_coordinates[second].group_number;
		}
		else
		if (rle_coordinates[second].group_number == 0 && rle_coordinates[first].group_number != 0)
		{
			rle_coordinates[second].group_number = rle_coordinates[first].group_number;
		}
		else
		if (rle_coordinates[second].group_number == 0 && rle_coordinates[first].group_number == 0)
		{
			number_of_groups++;
			rle_coordinates[first].group_number = number_of_groups;
			rle_coordinates[second].group_number = number_of_groups;


		}
		else
		{
			// This is the complicated bit.. if we got here then both are in the same group, but have different
			// group numbers, so we have to replace all occurences of the first with the last..

			long coord_counter;
			long group_to_replace;
			long group_to_replace_with;

			group_to_replace = rle_coordinates[second].group_number;
			group_to_replace_with = rle_coordinates[first].group_number;

			for (coord_counter = 0; coord_counter < number_of_coordinates; coord_counter++)
			{
				if (rle_coordinates[coord_counter].group_number == group_to_replace) rle_coordinates[coord_counter].group_number = group_to_replace_with;

			}

		}
	}

};
