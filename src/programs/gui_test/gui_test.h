class
        GuiTestApp : public wxApp {

  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

    static void TestCtfNodes(wxFrame* main_frame);

  private:
    bool ci_mode = false;
};

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
        {
                {wxCMD_LINE_SWITCH, "h", "help", "displays help on the command line parameters",
                 wxCMD_LINE_VAL_NONE, wxCMD_LINE_OPTION_HELP},

                {wxCMD_LINE_SWITCH, "c", "ci", "Run in continous integration mode"},

                {wxCMD_LINE_NONE}};

class GuiTestMainFrame : public wxFrame {
  public:
    GuiTestMainFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
    bool ci_mode = false;

  private:
    wxTimer m_timer;
    void    OnTimer(wxTimerEvent& event);
    //void OnHello(wxCommandEvent& event);
    //void OnExit(wxCommandEvent& event);
    //void OnAbout(wxCommandEvent& event);
    wxDECLARE_EVENT_TABLE( );
};
