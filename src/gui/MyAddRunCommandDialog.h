#ifndef __MyAddRunCommandDialog__
#define __MyAddRunCommandDialog__

/** Implementing AddRunCommandDialog */
class MyAddRunCommandDialog : public AddRunCommandDialog
{
	MyRunProfilesPanel *my_parent;

	public:



		/** Constructor */
		MyAddRunCommandDialog( MyRunProfilesPanel *parent );

		void ProcessResult();

		void OnOKClick( wxCommandEvent& event );
		void OnCancelClick( wxCommandEvent& event );
		void OnEnter( wxCommandEvent& event );
		void OnOverrideCheckbox( wxCommandEvent& event );

	//// end generated class members
	
};

#endif // __MyAddRunCommandDialog__
