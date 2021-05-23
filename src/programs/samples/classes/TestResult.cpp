#include "TestResult.hpp"

TestResult::TestResult()
{
    // Nothing to do
}

TestResult::~TestResult()
{
    // Nothing to do
}


void TestResult::PrintResults(wxString testName, bool& result) {

    testName = indendation + testName + " test";
    result ? wxPrintf("\t[Success] : ") : wxPrintf("\t [Failed] : ");

    wxPrintf("  %s\n", testName);


    allPassed &= result;
    result = true;
}