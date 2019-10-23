// <class_header>
// <class>ProgressBar</class>
//
// <author>Timothy Grant</author>
//
// <date>20090109</date>
//
// <description>
//
// A Simple Class to show a basic command line progress bar.  It
// includes the total percent complete, and also a self estimated
// Time Remaining..
//
// To use, simply call the constructor with the "number_of_ticks"
// required.  For example when dealing with images this would
// typically be the total number of images - although it can of
// course be anything.
//
// Once the ProgressBar object is constructed it will be printed
// to the console in it's initialised form (it is important that
// you output no other text during the lifetime of the progress bar
// or everything will look a bit weird.  In order to update the
// progress bar you must call the Update method with the tick
// you are currently on (in the above example this would be the
// image you are currently working on).
//
// After finishing you should delete the progress bar, which erases it.
//
// NOTE : If you call the status bar with one tick it will do nothing.
// I implemented this for sanity as it is stupid to have a progress
// bar for 1 image or 1 anything really.  However rather than write
// code in every program that checks how many images there are before
// creating a progress bar, I thought it would be easier to just do
// the check here.
//
// A Quick Example, meanfiltering a bunch of images :-
//
// void main(void)
// {
//    TigrisImage my_image("my_file", 1);
//    long number_of_images = my_image.number_following + 1;
//
//    ProgressBar *my_progress_bar = new ProgressBar(number_of_images);
//
//    for (long counter = 1; counter <= number_of_images; counter++)
//    {
//  	  input_image.Read(input_filename, counter);
//  	  input_image.MeanFilter(filter_size);
//  	  input_image.Write(output_filename, counter);
//  	  my_progress_bar->Update(counter);
//    }
//
//    delete my_progress_bar;
// }
//
// </class_header>

class ProgressBar {

private:

	long total_number_of_ticks;
	long start_time;
	long last_update_time;
	bool limit_to_100_percent;

public :

	// Constructors

	ProgressBar();
	ProgressBar(long wanted_total_number_of_ticks, bool wanted_limit_to_100_percent = true);

	// Destructor

	~ProgressBar();

	// Methods

	virtual void Update(long current_tick);
	virtual void CallOnUpdate() {}; // function that can be overidden to have control on extra things passed progressbars should do (this is primarily used for allowing GUI panels to track progress bars


};
