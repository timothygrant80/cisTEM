#ifndef TESTRESULT_HPP_
#define TESTRESULT_HPP_

#include "../../../core/core_headers.h"

class TestResult {

    public: 
         TestResult();
        ~TestResult();

        void PrintResults(wxString testName, bool& result);
        inline bool ReturnAllPassed() {return allPassed;};



    private:
        wxString indendation = "  ";
        bool allPassed = true;

};

#endif

