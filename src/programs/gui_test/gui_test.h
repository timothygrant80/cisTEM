class
        GuiTestApp : public wxApp {

  public:
    virtual bool OnInit( );
    virtual int  OnExit( );

    static void TestCtfNodes(wxFrame* main_frame);
};
